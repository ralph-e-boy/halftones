"""
Halftone Visualizer
-------------------
An interactive Streamlit app that demonstrates the major halftone screening
techniques described in the Getty "Atlas of Analytical Signatures of
Photographic Processes — Halftone" (Stulik & Kaplan, 2013) plus modern
digital halftoning algorithms.

Implemented techniques:
  - Classical (Ives/Levy) AM dot screen with selectable dot shape,
    screen ruling (LPI) and screen angle
  - CMYK process color (four-screen) with classic newspaper angles
  - Line screen (Akrography) and wavy-line screen
  - Stochastic / FM screen (grain Autotype style)
  - Metzograph (cracked-resist) and Erwin (reticulated gelatin) grain
  - Bayer ordered dithering (modern, deterministic)
  - Floyd-Steinberg error diffusion (modern, error-propagating)

Each renderer takes a grayscale (or RGB) intensity field in [0,1] and
returns a print-on-paper RGB image.
"""

from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Halftone", layout="wide",
                   initial_sidebar_state="expanded")

# Aggressively compact layout: trim block padding, shrink labels, tighten
# every widget's vertical gap so the halftone owns the viewport.
st.markdown(
    """
    <style>
      .block-container { padding: 0.4rem 0.8rem 0.4rem 0.8rem !important;
                         max-width: 100% !important; }
      section[data-testid="stSidebar"] .block-container {
          padding: 0.5rem 0.6rem !important; }
      section[data-testid="stSidebar"] { min-width: 240px !important;
                                          max-width: 260px !important; }
      h1, h2, h3, h4 { margin: 0.1rem 0 !important; padding: 0 !important;
                       font-size: 0.85rem !important; }
      /* The image container is sized to the viewport (minus chrome) and
         the img inside scales to fill it while preserving aspect ratio. */
      div[data-testid="stImage"] {
          display: flex;
          justify-content: center;
          align-items: center;
          width: 100%;
          height: 88vh;
      }
      div[data-testid="stImage"] img {
          border-radius: 3px;
          width: 100%;
          height: 100%;
          object-fit: contain;
      }
      /* Tighten every widget block. */
      div[data-testid="stVerticalBlock"] { gap: 0.25rem !important; }
      div[data-testid="stHorizontalBlock"] { gap: 0.35rem !important; }
      /* Smaller widget labels. */
      label, .stMarkdown p { font-size: 0.72rem !important;
                              margin-bottom: 0 !important; }
      /* Slim slider track. */
      div[data-baseweb="slider"] { padding: 0 0.4rem !important; }
      /* Color picker swatch — compact. */
      div[data-testid="stColorPicker"] > div { padding: 0 !important; }
      /* Segmented control buttons tighter. */
      button[kind="pillsActive"], button[kind="pills"],
      button[kind="segmentedControlActive"], button[kind="segmentedControl"] {
          padding: 0.15rem 0.45rem !important; font-size: 0.72rem !important; }
      /* Selectbox & inputs slimmer. */
      div[data-baseweb="select"] > div { min-height: 28px !important; }
      div[data-baseweb="input"] input { padding: 2px 6px !important; }
      /* Hide deploy chrome to recover vertical space. */
      header[data-testid="stHeader"] { height: 0; visibility: hidden; }
      /* Remove the gap above the first element. */
      div[data-testid="stAppViewBlockContainer"] > div:first-child {
          padding-top: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Source images
# ---------------------------------------------------------------------------

def make_linear_gradient(size: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.tile(x, (size, 1))


def make_radial_gradient(size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = (size - 1) * 0.5
    r = np.hypot(x - cx, y - cy) / (size * 0.5)
    return np.clip(1.0 - r, 0.0, 1.0)


def make_step_wedge(size: int, steps: int = 11) -> np.ndarray:
    out = np.empty((size, size), dtype=np.float32)
    band = size // steps
    for i in range(steps):
        v = i / (steps - 1)
        out[:, i * band:(i + 1) * band] = v
    out[:, steps * band:] = 1.0
    return out


def make_shaded_sphere(size: int) -> np.ndarray:
    """A diffuse-shaded sphere sitting on a graduated ground plane with a
    soft cast shadow. Exercises the full tonal range with smooth gradients,
    so dot growth from highlight → midtone → shadow is easy to see."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size * 0.5, size * 0.55
    r = size * 0.32
    # Sphere mask + surface normal (z out of the plane).
    dx, dy = (x - cx) / r, (y - cy) / r
    inside = dx * dx + dy * dy
    nz = np.sqrt(np.clip(1.0 - inside, 0.0, 1.0))
    mask = inside <= 1.0
    # Lambertian shading with the light from upper-left + a bright specular.
    L = np.array([-0.55, -0.55, 0.63], dtype=np.float32)
    L /= np.linalg.norm(L)
    ndotl = np.clip(dx * L[0] + dy * L[1] + nz * L[2], 0.0, 1.0)
    diffuse = 0.18 + 0.82 * ndotl
    # Specular: narrow highlight where the half-vector aligns with the normal.
    H = np.array([-0.27, -0.27, 0.92], dtype=np.float32)
    H /= np.linalg.norm(H)
    ndoth = np.clip(dx * H[0] + dy * H[1] + nz * H[2], 0.0, 1.0)
    specular = ndoth ** 24
    sphere = np.clip(diffuse + 0.55 * specular, 0.0, 1.0)
    # Background: top-bright sky-ish gradient.
    bg = 0.96 - 0.55 * (y / size)
    # Cast shadow — an offset ellipse, softly attenuated.
    sx = (x - (cx + 0.18 * size)) / (r * 1.05)
    sy = (y - (cy + 0.55 * r)) / (r * 0.28)
    shadow = np.clip(1.0 - (sx * sx + sy * sy), 0.0, 1.0) ** 0.7
    bg = bg * (1.0 - 0.55 * shadow)
    img = np.where(mask, sphere, bg)
    return np.clip(img.astype(np.float32), 0.0, 1.0)


APPLE_PHOTO_PATH = os.path.join(APP_DIR, "apple.png")


def make_apple_still_life(size: int) -> np.ndarray:
    """Real apple-still-life photo bundled with the app. Loaded once and
    cached so resizing the working size is cheap."""
    return _load_apple_rgb(size)


@st.cache_data(show_spinner=False)
def _load_apple_rgb(size: int) -> np.ndarray:
    img = Image.open(APPLE_PHOTO_PATH).convert("RGB")
    iw, ih = img.size
    scale = size / max(iw, ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_color_test(size: int) -> np.ndarray:
    """A color test card: a hue strip across the top, a saturation/value
    ramp body, and a few primary swatches. Designed to show off CMYK
    separation and rosette patterns without needing an upload."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    u, v = x / size, y / size
    # Base: HSV ramp (hue along x, value along y).
    h = u
    s = np.where(v < 0.65, 1.0, 1.0 - (v - 0.65) / 0.35)
    val = np.where(v < 0.15, 1.0, np.where(v < 0.65, 1.0, 1.0))
    val = np.clip(val, 0.0, 1.0)
    # HSV -> RGB.
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = val * (1 - s)
    q = val * (1 - f * s)
    t = val * (1 - (1 - f) * s)
    r = np.choose(i, [val, q, p, p, t, val])
    g = np.choose(i, [t, val, val, q, p, p])
    b = np.choose(i, [p, p, t, val, val, q])
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    # Top strip: pure hue, no shading.
    rgb[:int(size * 0.12)] = np.stack([r[:int(size * 0.12)],
                                         g[:int(size * 0.12)],
                                         b[:int(size * 0.12)]], axis=-1)
    # Three primary CMYK swatches at the bottom.
    band_y0, band_y1 = int(size * 0.82), int(size * 0.96)
    swatches = [(0.05, 0.25, (0.0, 1.0, 1.0)),  # C
                (0.30, 0.50, (1.0, 0.0, 1.0)),  # M
                (0.55, 0.75, (1.0, 1.0, 0.0)),  # Y
                (0.80, 0.95, (0.0, 0.0, 0.0))] # K
    for x0f, x1f, col in swatches:
        rgb[band_y0:band_y1, int(x0f * size):int(x1f * size)] = col
    return np.clip(rgb, 0.0, 1.0)


def load_user_image(file, size: int) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    # Rec. 709 luminance.
    return (0.2126 * img[..., 0]
            + 0.7152 * img[..., 1]
            + 0.0722 * img[..., 2]).astype(np.float32)


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def hex_to_rgb01(hex_str: str) -> np.ndarray:
    s = hex_str.lstrip("#")
    return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)],
                    dtype=np.float32)


def ink_over_paper(coverage: np.ndarray,
                   ink_rgb: np.ndarray,
                   paper_rgb: np.ndarray) -> np.ndarray:
    """Composite an ink coverage mask in [0,1] over a paper background.
    `coverage` may be a binary 0/1 mask or a fractional coverage."""
    c = coverage[..., None]
    return paper_rgb[None, None, :] * (1.0 - c) + ink_rgb[None, None, :] * c


# ---------------------------------------------------------------------------
# Halftone primitives
# ---------------------------------------------------------------------------

def _rotated_grid(shape: tuple[int, int],
                  cell_px: float,
                  angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) coordinates of every output pixel in a cell-local frame
    rotated by `angle_deg`. u and v are in units of cells; their fractional
    parts give the within-cell coordinates [0,1)."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    u = (c * x + s * y) / cell_px
    v = (-s * x + c * y) / cell_px
    return u, v


def _dot_distance(uf: np.ndarray, vf: np.ndarray, shape: str) -> np.ndarray:
    """Distance-from-cell-center metric scaled so 0 = center and ~1 = edge of
    the largest dot that fits in a cell."""
    du = uf - 0.5
    dv = vf - 0.5
    if shape == "round":
        # 0 at center, 1 at the corner sqrt(0.5) — scale so that a "100%
        # coverage" dot reaches the cell corners.
        return np.hypot(du, dv) / math.sqrt(0.5)
    if shape == "square":
        return np.maximum(np.abs(du), np.abs(dv)) / 0.5
    if shape == "diamond":
        return (np.abs(du) + np.abs(dv)) / 0.5
    if shape == "elliptical":
        # Elliptical dot used in modern offset printing — bridges in midtones
        # along one diagonal first, then the other.
        return np.hypot(du * 1.0, dv * 0.65) / math.sqrt(0.5 * (1 + 0.65 ** 2))
    if shape == "line":
        return np.abs(dv) / 0.5
    raise ValueError(f"unknown dot shape: {shape}")


@dataclass
class AMParams:
    lpi: int = 60                 # halftone screen ruling (lines per inch)
    dpi: int = 300                # output resolution
    angle_deg: float = 45.0       # classical K-channel angle
    shape: str = "round"
    halo: float = 0.0             # letterpress ink halo strength
    jitter: float = 0.0           # positional jitter (frac. of cell)
    invert: bool = False          # treat input as negative
    tone_gamma: float = 1.0       # tone-curve exponent (dot-gain comp)
    min_dot: float = 0.0          # minimum dot radius — chokes pinholes
    max_dot: float = 1.0          # maximum dot radius — leaves white in dmax


def am_halftone(intensity: np.ndarray,
                p: AMParams,
                ink_rgb: np.ndarray,
                paper_rgb: np.ndarray) -> np.ndarray:
    """Classical Ives/Levy AM (amplitude-modulated) halftone screen.

    Intensity is in [0,1]; 1 = white paper, 0 = full ink coverage. The dot
    size at each pixel is set by the *local* intensity, so the screen follows
    the picture instead of being applied as a global threshold map."""
    h, w = intensity.shape
    cell_px = max(1.0, p.dpi / max(1, p.lpi))
    u, v = _rotated_grid(intensity.shape, cell_px, p.angle_deg)
    uf, vf = u - np.floor(u), v - np.floor(v)

    if p.jitter > 0.0:
        # Hash-based per-cell offsets keep neighbouring cells uncorrelated
        # without needing a separate RNG state per call.
        iu, iv = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
        rng = np.sin(iu * 12.9898 + iv * 78.233) * 43758.5453
        rng2 = np.sin(iu * 39.3468 + iv * 11.135) * 24634.6345
        uf = (uf + (rng - np.floor(rng) - 0.5) * p.jitter) % 1.0
        vf = (vf + (rng2 - np.floor(rng2) - 0.5) * p.jitter) % 1.0

    d = _dot_distance(uf, vf, p.shape)
    img = intensity if not p.invert else 1.0 - intensity
    img = np.clip(img, 0.0, 1.0)
    # Tone curve: gamma > 1 darkens midtones (counteracts dot gain on
    # absorbent paper); gamma < 1 lightens them.
    if p.tone_gamma != 1.0:
        img = img ** (1.0 / max(0.05, p.tone_gamma))
    # Map intensity to target radius (area-preserving for round dots), then
    # clamp the printable radius range.
    target = np.sqrt(1.0 - img)
    target = np.clip(target, p.min_dot, p.max_dot)
    coverage = (d < target).astype(np.float32)

    if p.halo > 0.0:
        # Letterpress halo: a slightly larger ring of darker ink around each
        # dot with a lightened centre, as described in the Getty atlas.
        ring = ((d < target + 0.18) & (d > target * 0.55)).astype(np.float32)
        center = (d < target * 0.45).astype(np.float32)
        coverage = np.clip(coverage + p.halo * ring - 0.5 * p.halo * center,
                           0.0, 1.0)

    return ink_over_paper(coverage, ink_rgb, paper_rgb)


# ---------------------------------------------------------------------------
# CMYK process color
# ---------------------------------------------------------------------------

# Classical newspaper screen angles. Yellow is offset by 15° from black to
# minimise moiré; cyan and magenta sit symmetrically on either side.
CLASSIC_ANGLES = {"C": 15.0, "M": 75.0, "Y": 0.0, "K": 45.0}
INK_COLORS = {
    "C": np.array([0.0, 1.0, 1.0], dtype=np.float32),
    "M": np.array([1.0, 0.0, 1.0], dtype=np.float32),
    "Y": np.array([1.0, 1.0, 0.0], dtype=np.float32),
    "K": np.array([0.0, 0.0, 0.0], dtype=np.float32),
}


def rgb_to_cmyk(rgb: np.ndarray) -> dict[str, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    k = 1.0 - np.maximum.reduce([r, g, b])
    denom = np.where(k < 1.0, 1.0 - k, 1.0)
    c = (1.0 - r - k) / denom
    m = (1.0 - g - k) / denom
    y = (1.0 - b - k) / denom
    return {"C": np.clip(c, 0.0, 1.0),
            "M": np.clip(m, 0.0, 1.0),
            "Y": np.clip(y, 0.0, 1.0),
            "K": np.clip(k, 0.0, 1.0)}


def cmyk_halftone(rgb_img: np.ndarray,
                  p: AMParams,
                  angles: dict[str, float],
                  paper_rgb: np.ndarray,
                  channels: tuple[str, ...] = ("C", "M", "Y", "K"),
                  dot_gains: dict[str, float] = None,
                  ) -> np.ndarray:
    """Print each separation as its own AM screen and composite using a
    subtractive ink model (multiply). dot_gains dict provides per-channel
    dot gain compensation (0-30% typical range)."""
    if dot_gains is None:
        dot_gains = {ch: 0.0 for ch in "CMYK"}

    seps = rgb_to_cmyk(rgb_img if rgb_img.ndim == 3 else
                       np.stack([rgb_img] * 3, axis=-1))
    target_shape = rgb_img.shape if rgb_img.ndim == 3 else (*rgb_img.shape, 3)
    out = np.broadcast_to(paper_rgb, target_shape).astype(np.float32).copy()
    for ch in channels:
        ink = INK_COLORS[ch]
        # Apply tone curve correction for dot gain on this channel
        sep = 1.0 - seps[ch]
        gain = dot_gains.get(ch, 0.0)
        if gain > 0:
            # Convert dot gain % to tone gamma: higher gain -> lower gamma (darken)
            tone_gamma = max(0.3, 1.0 - (gain / 100.0) * 0.4)
            sep = sep ** (1.0 / max(0.05, tone_gamma))

        # Screen the separation against a white background to recover a clean
        # binary coverage mask for this ink.
        plane = am_halftone(sep,
                            AMParams(lpi=p.lpi, dpi=p.dpi,
                                     angle_deg=angles[ch],
                                     shape=p.shape, halo=p.halo,
                                     jitter=p.jitter),
                            ink_rgb=ink, paper_rgb=np.ones(3, dtype=np.float32))
        # `plane` is white where there's no ink and `ink` where there is, so
        # it behaves as a subtractive multiplier. Stacking inks via multiply
        # is the standard simple ink model (Beer-Lambert-ish).
        out = out * plane
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Line and wavy-line screens
# ---------------------------------------------------------------------------

def line_screen(intensity: np.ndarray,
                lpi: int, dpi: int, angle_deg: float,
                ink_rgb: np.ndarray, paper_rgb: np.ndarray,
                crosshatch: bool = False,
                soft_edge: float = 0.0) -> np.ndarray:
    """Akrography-style line screen: parallel ruled lines whose thickness is
    modulated by image intensity. Optionally adds a perpendicular second set
    of lines (crosshatch) and/or a soft anti-aliased edge."""
    cell_px = max(1.0, dpi / max(1, lpi))
    _, v = _rotated_grid(intensity.shape, cell_px, angle_deg)
    vf = v - np.floor(v)
    target = 1.0 - np.clip(intensity, 0.0, 1.0)

    def _cov(d: np.ndarray, t: np.ndarray) -> np.ndarray:
        if soft_edge <= 0.0:
            return (d < t * 0.5).astype(np.float32)
        # Smooth ramp of width `soft_edge` (in normalised cell units).
        return np.clip((t * 0.5 - d) / max(1e-3, soft_edge) + 0.5, 0.0, 1.0)

    coverage = _cov(np.abs(vf - 0.5), target)
    if crosshatch:
        _, v2 = _rotated_grid(intensity.shape, cell_px, angle_deg + 90.0)
        vf2 = v2 - np.floor(v2)
        # For crosshatch, each ruling carries half the tone so the two
        # rulings sum to the right total coverage at midtones.
        c2 = _cov(np.abs(vf2 - 0.5), target * 0.6)
        coverage = np.clip(coverage + c2 - coverage * c2, 0.0, 1.0)
    return ink_over_paper(coverage, ink_rgb, paper_rgb)


def wavy_line_screen(intensity: np.ndarray,
                     lpi: int, dpi: int, angle_deg: float,
                     wave_amp: float, wave_freq: float,
                     ink_rgb: np.ndarray,
                     paper_rgb: np.ndarray) -> np.ndarray:
    """Wavy-line screen: same as line screen but each line follows a
    sinusoid. `wave_amp` is in cell widths, `wave_freq` in cycles per cell."""
    cell_px = max(1.0, dpi / max(1, lpi))
    u, v = _rotated_grid(intensity.shape, cell_px, angle_deg)
    phase = np.sin(2.0 * math.pi * wave_freq * u) * wave_amp
    v_shifted = v + phase
    vf = v_shifted - np.floor(v_shifted)
    target = 1.0 - np.clip(intensity, 0.0, 1.0)
    coverage = (np.abs(vf - 0.5) < target * 0.5).astype(np.float32)
    return ink_over_paper(coverage, ink_rgb, paper_rgb)


# ---------------------------------------------------------------------------
# Stochastic / grain screens
# ---------------------------------------------------------------------------

def stochastic_screen(intensity: np.ndarray,
                      cell_px: float,
                      ink_rgb: np.ndarray,
                      paper_rgb: np.ndarray,
                      seed: int = 0,
                      hi_freq: float = 0.3) -> np.ndarray:
    """FM (frequency-modulated) screen — fixed-size random dots with density
    proportional to ink amount. Visually similar to the grain Autotype
    process. Mixes a grain-scale noise field with a pixel-scale dither;
    `hi_freq` controls the dither weight."""
    rng = np.random.default_rng(seed)
    h, w = intensity.shape
    nh = max(2, int(round(h / cell_px)))
    nw = max(2, int(round(w / cell_px)))
    noise_lo = rng.random((nh, nw), dtype=np.float32)
    noise = zoom(noise_lo, (h / nh, w / nw), order=1)[:h, :w]
    hf = np.clip(hi_freq, 0.0, 1.0)
    noise = (1.0 - hf) * noise + hf * rng.random((h, w), dtype=np.float32)
    coverage = (noise > intensity).astype(np.float32)
    return ink_over_paper(coverage, ink_rgb, paper_rgb)


def metzograph_screen(intensity: np.ndarray,
                      cell_px: float,
                      crack_density: float,
                      ink_rgb: np.ndarray,
                      paper_rgb: np.ndarray,
                      seed: int = 0) -> np.ndarray:
    """Metzograph-style screen: a high-frequency irregular pattern produced
    by thresholding low-pass-filtered noise (mimicking the cracked-resist
    pattern etched into the glass screen)."""
    rng = np.random.default_rng(seed)
    h, w = intensity.shape
    n = rng.standard_normal((h, w)).astype(np.float32)
    sigma = max(0.5, cell_px * 0.6)
    smooth = gaussian_filter(n, sigma=sigma)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)
    # Add a thin random "cracks" component.
    cracks = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32),
                             sigma=max(0.4, cell_px * 0.25))
    cracks = (cracks - cracks.min()) / (cracks.max() - cracks.min() + 1e-9)
    screen = (1.0 - crack_density) * smooth + crack_density * cracks
    coverage = (screen > intensity).astype(np.float32)
    return ink_over_paper(coverage, ink_rgb, paper_rgb)


def erwin_grain_screen(intensity: np.ndarray,
                       grain_px: float,
                       reticulation: float,
                       ink_rgb: np.ndarray,
                       paper_rgb: np.ndarray,
                       seed: int = 0) -> np.ndarray:
    """Erwin reticulated-gelatin grain screen — long, sinuous filaments. We
    approximate the reticulation by warping a sin/cos grid through two
    smooth-noise fields. The noise is generated at coarse resolution
    (≈ grain_px subsample) and bilinearly upscaled — the visible feature
    size is set by `grain_px * 1.2`, so finer detail than that would be
    blurred away anyway."""
    rng = np.random.default_rng(seed)
    h, w = intensity.shape
    k = max(2.0, float(grain_px))
    ch, cw = max(4, int(round(h / k))), max(4, int(round(w / k)))
    n1c = gaussian_filter(rng.standard_normal((ch, cw)).astype(np.float32),
                          sigma=1.2)
    n2c = gaussian_filter(rng.standard_normal((ch, cw)).astype(np.float32),
                          sigma=1.2)
    n1 = zoom(n1c, (h / ch, w / cw), order=1)[:h, :w]
    n2 = zoom(n2c, (h / ch, w / cw), order=1)[:h, :w]
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    warp = reticulation * grain_px * 3.0
    sx = xs + warp * (n1 - n1.mean()) / (n1.std() + 1e-9)
    sy = ys + warp * (n2 - n2.mean()) / (n2.std() + 1e-9)
    freq = 2.0 * math.pi / max(1.0, grain_px * 1.4)
    base = np.sin(freq * sx) * np.cos(freq * sy)
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    coverage = (base > intensity).astype(np.float32)
    return ink_over_paper(coverage, ink_rgb, paper_rgb)


# ---------------------------------------------------------------------------
# Modern digital halftoning
# ---------------------------------------------------------------------------

def bayer_matrix(n: int) -> np.ndarray:
    """Return an n x n Bayer ordered-dither matrix normalised to [0,1).
    `n` must be a power of two. Built in integer form so the recursion is
    obviously correct, then normalised once at the end."""
    def _int(k: int) -> np.ndarray:
        if k == 1:
            return np.array([[0]], dtype=np.int32)
        sub = _int(k // 2)
        top = np.hstack([4 * sub + 0, 4 * sub + 2])
        bot = np.hstack([4 * sub + 3, 4 * sub + 1])
        return np.vstack([top, bot])
    return (_int(n).astype(np.float32) + 0.5) / (n * n)


def bayer_dither(intensity: np.ndarray, n: int,
                 ink_rgb: np.ndarray, paper_rgb: np.ndarray) -> np.ndarray:
    m = bayer_matrix(n)
    h, w = intensity.shape
    tile = np.tile(m, (h // n + 1, w // n + 1))[:h, :w]
    coverage = (intensity < tile).astype(np.float32)
    return ink_over_paper(coverage, ink_rgb, paper_rgb)


def floyd_steinberg(intensity: np.ndarray,
                    ink_rgb: np.ndarray,
                    paper_rgb: np.ndarray) -> np.ndarray:
    """Floyd–Steinberg via PIL's C path — fast default for the classic
    kernel. Other kernels go through `error_diffuse` below."""
    pil = Image.fromarray((np.clip(intensity, 0.0, 1.0) * 255.0)
                          .astype(np.uint8), mode="L")
    dithered = np.asarray(pil.convert("1", dither=Image.FLOYDSTEINBERG),
                          dtype=np.float32)
    coverage = 1.0 - dithered
    return ink_over_paper(coverage, ink_rgb, paper_rgb)


# Each kernel: (offsets, weights). Offsets are (dy, dx) relative to the
# current pixel; weights are integer numerators with a shared denominator
# given as the third entry. Only forward (un-processed) neighbours appear.
ERROR_KERNELS: dict[str, tuple[list[tuple[int, int]], list[int], int]] = {
    "Floyd–Steinberg": ([(0, 1), (1, -1), (1, 0), (1, 1)],
                        [7, 3, 5, 1], 16),
    "Atkinson":        ([(0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)],
                        [1, 1, 1, 1, 1, 1], 8),  # 6/8 — drops some error
    "Jarvis-J-N":      ([(0, 1), (0, 2),
                          (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
                          (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)],
                         [7, 5, 3, 5, 7, 5, 3, 1, 3, 5, 3, 1], 48),
    "Stucki":          ([(0, 1), (0, 2),
                          (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
                          (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)],
                         [8, 4, 2, 4, 8, 4, 2, 1, 2, 4, 2, 1], 42),
    "Sierra-3":        ([(0, 1), (0, 2),
                          (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
                          (2, -1), (2, 0), (2, 1)],
                         [5, 3, 2, 4, 5, 4, 2, 2, 3, 2], 32),
    "Burkes":          ([(0, 1), (0, 2),
                          (1, -2), (1, -1), (1, 0), (1, 1), (1, 2)],
                         [8, 4, 2, 4, 8, 4, 2], 32),
}


def error_diffuse(intensity: np.ndarray,
                  kernel: str,
                  serpentine: bool,
                  ink_rgb: np.ndarray,
                  paper_rgb: np.ndarray) -> np.ndarray:
    """Generic error-diffusion driver. Sequential by nature — Python loop
    over rows, vectorised within a row. Pure-Python error diffusion is
    inherently O(N²); keep the working size modest (≤512) for snappy
    interaction."""
    offsets, weights, denom = ERROR_KERNELS[kernel]
    img = intensity.astype(np.float32).copy()
    h, w = img.shape
    inv_denom = 1.0 / denom
    for y in range(h):
        ltr = (y % 2 == 0) or (not serpentine)
        row_iter = range(w) if ltr else range(w - 1, -1, -1)
        for x in row_iter:
            old = img[y, x]
            new = 1.0 if old > 0.5 else 0.0
            img[y, x] = new
            err = old - new
            for (dy, dx), wt in zip(offsets, weights):
                # Mirror dx for serpentine right-to-left passes.
                ddx = dx if ltr else -dx
                yy, xx = y + dy, x + ddx
                if 0 <= yy < h and 0 <= xx < w:
                    img[yy, xx] += err * wt * inv_denom
    coverage = 1.0 - np.clip(img, 0.0, 1.0)
    return ink_over_paper(coverage, ink_rgb, paper_rgb)


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray, *,
               gamma: float = 1.0,
               brightness: float = 0.0,
               contrast: float = 1.0,
               blur_sigma: float = 0.0,
               sharpen: float = 0.0,
               black_point: float = 0.0,
               white_point: float = 1.0,
               mid_point: float = 0.5,
               saturation: float = 1.0,
               posterize: int = 0,
               vignette: float = 0.0,
               threshold: float = 0.0) -> np.ndarray:
    """Apply tonal / spatial adjustments to the source image before the
    halftone screen sees it. Works on grayscale or RGB. Order of operations:
    spatial (blur, sharpen) -> levels -> gamma -> contrast -> brightness ->
    saturation -> posterize -> vignette -> threshold."""
    out = img.astype(np.float32)

    if blur_sigma > 0.0:
        if out.ndim == 3:
            for c in range(out.shape[-1]):
                out[..., c] = gaussian_filter(out[..., c], sigma=blur_sigma)
        else:
            out = gaussian_filter(out, sigma=blur_sigma)
    if sharpen > 0.0:
        if out.ndim == 3:
            blurred = np.empty_like(out)
            for c in range(out.shape[-1]):
                blurred[..., c] = gaussian_filter(out[..., c], sigma=1.5)
        else:
            blurred = gaussian_filter(out, sigma=1.5)
        out = out + sharpen * (out - blurred)

    # Photoshop-style levels: linear remap of [bp, wp] -> [0, 1] then a
    # midpoint gamma so the histogram pivots around `mid_point`.
    if black_point > 0.0 or white_point < 1.0 or mid_point != 0.5:
        bp = float(black_point)
        wp = max(bp + 1e-3, float(white_point))
        out = (out - bp) / (wp - bp)
        out = np.clip(out, 0.0, 1.0)
        # mid_point in [0,1]; gamma derived so input=mid_point -> output=0.5.
        if mid_point != 0.5:
            mp = float(np.clip(mid_point, 0.02, 0.98))
            g = math.log(0.5) / math.log(mp)
            out = out ** g

    if gamma != 1.0:
        out = np.clip(out, 0.0, 1.0) ** (1.0 / max(0.05, gamma))
    if contrast != 1.0:
        out = (out - 0.5) * contrast + 0.5
    if brightness != 0.0:
        out = out + brightness

    out = np.clip(out, 0.0, 1.0)

    if out.ndim == 3 and saturation != 1.0:
        # Desat / supersat around luminance.
        lum = (0.2126 * out[..., 0] + 0.7152 * out[..., 1]
               + 0.0722 * out[..., 2])[..., None]
        out = lum + (out - lum) * saturation
        out = np.clip(out, 0.0, 1.0)

    if posterize and posterize > 1:
        out = np.round(out * (posterize - 1)) / (posterize - 1)

    if vignette > 0.0:
        h, w = out.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        r = np.hypot(x - cx, y - cy) / max(1.0, math.hypot(cx, cy))
        # Smooth attenuation: 1 at centre, (1 - vignette) at corners.
        vmask = 1.0 - vignette * np.clip((r - 0.4) / 0.6, 0.0, 1.0) ** 2
        if out.ndim == 3:
            out = out * vmask[..., None]
        else:
            out = out * vmask

    if threshold > 0.0:
        # Hard binarise — useful for line-art prep.
        out = (out >= threshold).astype(np.float32)

    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Paper textures
# ---------------------------------------------------------------------------

PAPER_KINDS = ["none", "smooth coated", "newsprint", "rough uncoated",
               "linen", "kraft", "watercolor", "vintage", "foxed",
               "onion skin", "print-through"]


@st.cache_data(show_spinner=False)
def paper_texture(kind: str, shape: tuple[int, int],
                  strength: float) -> np.ndarray:
    """Return a multiplicative paper-texture map in [0,1]: 1 = clean, lower
    values darken. `kind` selects the texture style."""
    if kind == "none" or strength <= 0.0:
        return np.ones(shape, dtype=np.float32)
    rng = np.random.default_rng(7)
    h, w = shape
    n = rng.standard_normal((h, w)).astype(np.float32)

    if kind == "smooth coated":
        tex = gaussian_filter(n, sigma=2.5)
    elif kind == "newsprint":
        tex = (gaussian_filter(n, sigma=1.0)
               + 0.4 * gaussian_filter(rng.standard_normal((h, w))
                                       .astype(np.float32), sigma=4.0))
    elif kind == "rough uncoated":
        tex = (gaussian_filter(n, sigma=0.8)
               + 0.6 * gaussian_filter(rng.standard_normal((h, w))
                                       .astype(np.float32), sigma=2.0))
    elif kind == "linen":
        # Crosshatched fibre pattern: two sin gratings + grain.
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        warp = 0.15 * gaussian_filter(rng.standard_normal((h, w))
                                       .astype(np.float32), sigma=3.0)
        tex = (0.5 + 0.25 * np.sin(2 * math.pi * (x / 8.0 + warp))
               + 0.25 * np.sin(2 * math.pi * (y / 8.0 - warp))
               + 0.15 * gaussian_filter(n, sigma=0.6))
    elif kind == "kraft":
        # Long, directional fibres + faint speckle.
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        fibres = gaussian_filter(rng.standard_normal((h, w))
                                  .astype(np.float32), sigma=(0.8, 6.0))
        speck = (rng.random((h, w), dtype=np.float32) > 0.992).astype(np.float32)
        tex = 0.7 * fibres + 0.3 * gaussian_filter(n, sigma=0.6) - 0.4 * speck
    elif kind == "watercolor":
        # Coarse cold-press texture.
        tex = (gaussian_filter(n, sigma=0.4)
               + 0.8 * gaussian_filter(rng.standard_normal((h, w))
                                       .astype(np.float32), sigma=3.5)
               + 0.4 * gaussian_filter(rng.standard_normal((h, w))
                                       .astype(np.float32), sigma=8.0))
    elif kind == "vintage":
        big = gaussian_filter(rng.standard_normal((h, w))
                              .astype(np.float32), sigma=40.0)
        med = gaussian_filter(rng.standard_normal((h, w))
                              .astype(np.float32), sigma=8.0)
        fine = gaussian_filter(n, sigma=0.7)
        tex = 0.6 * big + 0.3 * med + 0.1 * fine
    elif kind == "foxed":
        # Vintage stains + scattered dark "foxing" spots.
        big = gaussian_filter(rng.standard_normal((h, w))
                              .astype(np.float32), sigma=30.0)
        spots = np.zeros((h, w), dtype=np.float32)
        n_spots = max(4, (h * w) // 8000)
        ys = rng.integers(0, h, size=n_spots)
        xs = rng.integers(0, w, size=n_spots)
        spots[ys, xs] = 1.0
        spots = gaussian_filter(spots, sigma=2.0)
        spots = (spots - spots.min()) / (spots.max() - spots.min() + 1e-9)
        tex = 0.55 * big + 0.45 * gaussian_filter(n, sigma=0.6) - 0.7 * spots
    elif kind == "onion skin":
        # Very faint, almost translucent.
        tex = (gaussian_filter(n, sigma=0.5)
               + 0.5 * gaussian_filter(rng.standard_normal((h, w))
                                       .astype(np.float32), sigma=1.5))
        # Compress dynamic range so the effect stays subtle.
        tex = tex * 0.4 + 0.3
    elif kind == "print-through":
        # Faint ghosted image / reversed text on the back side.
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        rows = ((np.sin(2 * math.pi * y / 9.0) > 0.0)
                * (rng.random((h, w), dtype=np.float32) > 0.55)
               ).astype(np.float32)
        rows = gaussian_filter(rows, sigma=0.7)
        tex = 0.6 * gaussian_filter(n, sigma=2.0) - 0.5 * rows
    else:
        tex = n

    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-9)
    return np.clip(1.0 - strength * (1.0 - tex), 0.0, 1.0)


def apply_paper(image_rgb: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """Multiply the printed image by a paper-texture map (broadcasts over
    color channels)."""
    return np.clip(image_rgb * texture[..., None], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Shader generation (GLSL fragment shader + Metal Shading Language)
# ---------------------------------------------------------------------------

# Modes that translate cleanly to a per-pixel GPU shader. Error diffusion is
# inherently sequential; Metzograph still needs a precomputed crack texture.
# Erwin uses value-noise warp (smoothed grid-hash), so it ports cleanly.
SHADER_MODES = {
    "Ives/Levy (AM dot)",
    "CMYK process color",
    "Line screen (Akrography)",
    "Wavy-line screen",
    "Stochastic / FM (Autotype)",
    "Bayer ordered dither",
    "Erwin grain (reticulated)",
}


def _shape_glsl(shape: str) -> str:
    """Return a GLSL expression that, given `du`/`dv` in [-0.5, 0.5], yields a
    normalised dot-distance metric (0 at centre, 1 at the cell corner)."""
    return {
        "round":      "length(vec2(du, dv)) / sqrt(0.5)",
        "square":     "max(abs(du), abs(dv)) / 0.5",
        "diamond":    "(abs(du) + abs(dv)) / 0.5",
        "elliptical": "length(vec2(du, dv*0.65)) / sqrt(0.5*(1.0+0.65*0.65))",
        "line":       "abs(dv) / 0.5",
    }[shape]


def _shape_metal(shape: str) -> str:
    return {
        "round":      "length(float2(du, dv)) / sqrt(0.5h)",
        "square":     "max(fabs(du), fabs(dv)) / 0.5",
        "diamond":    "(fabs(du) + fabs(dv)) / 0.5",
        "elliptical": "length(float2(du, dv*0.65)) / sqrt(0.5*(1.0+0.65*0.65))",
        "line":       "fabs(dv) / 0.5",
    }[shape]


def _glsl_aspect_correction() -> str:
    """GLSL code to handle aspect ratio and return scaled frag/uv coordinates."""
    return """\
    float canvas_aspect = u_resolution.x / u_resolution.y;
    float image_aspect = u_image_size.x / u_image_size.y;
    vec2 frag = v_uv * u_resolution;
    vec2 uv_tex = v_uv;
    bool in_bounds = true;
    if (canvas_aspect > image_aspect) {
        float img_w = u_resolution.y * image_aspect;
        float margin = (u_resolution.x - img_w) * 0.5;
        if (frag.x < margin || frag.x > margin + img_w) {
            in_bounds = false;
        } else {
            frag.x = (frag.x - margin) * u_image_size.x / img_w;
            frag.y = frag.y * u_image_size.y / u_resolution.y;
            uv_tex.x = (v_uv.x * u_resolution.x - margin) / img_w;
        }
    } else {
        float img_h = u_resolution.x / image_aspect;
        float margin = (u_resolution.y - img_h) * 0.5;
        if (frag.y < margin || frag.y > margin + img_h) {
            in_bounds = false;
        } else {
            frag.x = frag.x * u_image_size.x / u_resolution.x;
            frag.y = (frag.y - margin) * u_image_size.y / img_h;
            uv_tex.y = (v_uv.y * u_resolution.y - margin) / img_h;
        }
    }
    if (!in_bounds) {
        gl_FragColor = vec4(u_paper, 1.0);
        return;
    }
"""


GLSL_HEADER = """\
// Auto-generated by halftones/app.py — drop-in fragment shader.
// Inputs:
//   u_texture   — source image, sampled at v_uv
//   u_resolution — canvas size in pixels (vec2)
//   u_image_size — source image size in pixels (vec2)
//   u_ink, u_paper — print colours (vec3)
// Output: gl_FragColor (paper × ink composite).
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_texture;
uniform vec2 u_resolution;
uniform vec2 u_image_size;
uniform vec3 u_ink;
uniform vec3 u_paper;
"""

METAL_HEADER = """\
// Auto-generated by halftones/app.py — Metal fragment shader.
// Build with:  xcrun -sdk macosx metal -c halftone.metal -o halftone.air
//              xcrun -sdk macosx metallib halftone.air -o halftone.metallib
// Bind: texture(0)=source, sampler(0)=linear-clamp,
//       buffer(0)=float2 resolution, buffer(1)=float3 ink,
//       buffer(2)=float3 paper.
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 position [[position]];
    float2 uv;
};
"""


def _glsl_lum() -> str:
    return "dot(src.rgb, vec3(0.2126, 0.7152, 0.0722))"


def _glsl_am_body(p: dict) -> str:
    """Body of the AM dot main(), parameterised by params from the UI."""
    return f"""\
    {_glsl_aspect_correction()}
    vec4 src = texture2D(u_texture, uv_tex);
    float theta = radians({p['angle']:.4f});
    float c = cos(theta), s = sin(theta);
    vec2 rot = vec2(c * frag.x + s * frag.y, -s * frag.x + c * frag.y);
    float cell_px = {max(1.0, p['dpi'] / max(1, p['lpi'])):.4f};
    vec2 cf = fract(rot / cell_px);
    float lum = {_glsl_lum()};
    {('lum = 1.0 - lum;' if p.get('invert') else '')}
    lum = clamp(lum, 0.0, 1.0);
    float tone = {1.0 / max(0.05, p.get('tone', 1.0)):.4f};
    lum = pow(lum, tone);
    float target = clamp(sqrt(1.0 - lum), {p['dot_range'][0]:.4f}, {p['dot_range'][1]:.4f});
    float du = cf.x - 0.5, dv = cf.y - 0.5;
    float d = {_shape_glsl(p['shape'])};
    float coverage = step(d, target);
"""


def generate_glsl(mode: str, p: dict, ink: np.ndarray,
                  paper: np.ndarray) -> str:
    """Return a complete GLSL fragment shader string for the given mode."""
    if mode == "Ives/Levy (AM dot)":
        halo_block = ""
        if p.get("halo", 0.0) > 0.0:
            halo_block = f"""\
    float ring = step(d, target + 0.18) * (1.0 - step(d, target * 0.55));
    float ctr  = step(d, target * 0.45);
    coverage = clamp(coverage + {p['halo']:.4f} * ring
                              - 0.5 * {p['halo']:.4f} * ctr, 0.0, 1.0);
"""
        return f"""{GLSL_HEADER}
void main() {{
{_glsl_am_body(p)}{halo_block}
    gl_FragColor = vec4(mix(u_paper, u_ink, coverage), 1.0);
}}
"""

    if mode == "CMYK process color":
        ang = {"C": p["ang_c"], "M": p["ang_m"],
               "Y": p["ang_y"], "K": p["ang_k"]}
        cell = max(1.0, p['dpi'] / max(1, p['lpi']))
        shape_expr = _shape_glsl(p['shape'])
        # Inline the four-channel screening as a loop unroll.
        channel_block = ""
        ink_const = {
            "C": "vec3(0.0, 0.68, 0.94)",
            "M": "vec3(0.93, 0.0, 0.55)",
            "Y": "vec3(0.98, 0.93, 0.0)",
            "K": "vec3(0.05, 0.05, 0.05)",
        }
        chans = p.get("channels") or ["C", "M", "Y", "K"]
        for ch in chans:
            channel_block += f"""\
    {{
        float theta = radians({ang[f'ang_{ch.lower()}'] if False else ang[ch]:.4f});
        float c = cos(theta), s = sin(theta);
        vec2 rot = vec2(c * frag.x + s * frag.y, -s * frag.x + c * frag.y);
        vec2 cf = fract(rot / {cell:.4f});
        float target = sqrt(1.0 - clamp(sep_{ch.lower()}, 0.0, 1.0));
        float du = cf.x - 0.5, dv = cf.y - 0.5;
        float d = {shape_expr};
        float cov = step(d, target);
        out_rgb = out_rgb * mix(vec3(1.0), {ink_const[ch]}, cov);
    }}
"""
        return f"""{GLSL_HEADER}
void main() {{
    {_glsl_aspect_correction()}
    vec4 src = texture2D(u_texture, v_uv);
    // RGB -> CMYK separation
    float k = 1.0 - max(max(src.r, src.g), src.b);
    float denom = (k < 1.0) ? (1.0 - k) : 1.0;
    float sep_c = clamp((1.0 - src.r - k) / denom, 0.0, 1.0);
    float sep_m = clamp((1.0 - src.g - k) / denom, 0.0, 1.0);
    float sep_y = clamp((1.0 - src.b - k) / denom, 0.0, 1.0);
    float sep_k = clamp(k, 0.0, 1.0);
    vec3 out_rgb = u_paper;
{channel_block}
    gl_FragColor = vec4(out_rgb, 1.0);
}}
""".replace("texture2D(u_texture, v_uv)", "texture2D(u_texture, uv_tex)")

    if mode == "Line screen (Akrography)":
        cell = max(1.0, p['dpi'] / max(1, p['lpi']))
        cross = ""
        if p.get("crosshatch"):
            cross = f"""\
    {{
        float theta2 = radians({p['angle'] + 90.0:.4f});
        float c2 = cos(theta2), s2 = sin(theta2);
        float v2 = -s2 * frag.x + c2 * frag.y;
        float vf2 = fract(v2 / {cell:.4f});
        float t2 = target * 0.6;
        float c2_cov = step(abs(vf2 - 0.5), t2 * 0.5);
        coverage = clamp(coverage + c2_cov - coverage * c2_cov, 0.0, 1.0);
    }}
"""
        return f"""{GLSL_HEADER}
void main() {{
    {_glsl_aspect_correction()}
    float theta = radians({p['angle']:.4f});
    float c = cos(theta), s = sin(theta);
    float v = -s * frag.x + c * frag.y;
    float vf = fract(v / {cell:.4f});
    vec4 src = texture2D(u_texture, v_uv);
    float lum = {_glsl_lum()};
    float target = 1.0 - clamp(lum, 0.0, 1.0);
    float coverage = step(abs(vf - 0.5), target * 0.5);
{cross}
    gl_FragColor = vec4(mix(u_paper, u_ink, coverage), 1.0);
}}
""".replace("texture2D(u_texture, v_uv)", "texture2D(u_texture, uv_tex)")

    if mode == "Wavy-line screen":
        cell = max(1.0, p['dpi'] / max(1, p['lpi']))
        return f"""{GLSL_HEADER}
void main() {{
    {_glsl_aspect_correction()}
    float theta = radians({p['angle']:.4f});
    float c = cos(theta), s = sin(theta);
    float u = c * frag.x + s * frag.y;
    float v = -s * frag.x + c * frag.y;
    float u_n = u / {cell:.4f};
    float phase = sin(2.0 * 3.14159265 * {p['freq']:.4f} * u_n) * {p['amp']:.4f};
    float vf = fract(v / {cell:.4f} + phase);
    vec4 src = texture2D(u_texture, v_uv);
    float lum = {_glsl_lum()};
    float target = 1.0 - clamp(lum, 0.0, 1.0);
    float coverage = step(abs(vf - 0.5), target * 0.5);
    gl_FragColor = vec4(mix(u_paper, u_ink, coverage), 1.0);
}}
""".replace("texture2D(u_texture, v_uv)", "texture2D(u_texture, uv_tex)")

    if mode == "Stochastic / FM (Autotype)":
        # Hash-based pseudo-random noise.
        return f"""{GLSL_HEADER}
// Cheap GLSL hash: returns a pseudo-random float in [0,1) given a 2D seed.
float hash21(vec2 p) {{
    p = fract(p * vec2(443.8975, 397.2973));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}}
void main() {{
    {_glsl_aspect_correction()}
    float grain = {p['grain']:.4f};
    vec2 cell_xy = floor(frag / grain);
    float noise_lo = hash21(cell_xy);
    float noise_hi = hash21(frag);
    float noise = mix(noise_lo, noise_hi, {p['hi_freq']:.4f});
    vec4 src = texture2D(u_texture, v_uv);
    float lum = {_glsl_lum()};
    float coverage = step(lum, noise);
    gl_FragColor = vec4(mix(u_paper, u_ink, coverage), 1.0);
}}
""".replace("texture2D(u_texture, v_uv)", "texture2D(u_texture, uv_tex)")

    if mode == "Bayer ordered dither":
        n = int(p["n"])
        m = bayer_matrix(n)
        bias = p.get("thresh_bias", 0.0)
        # Generate individual assignments since GLSL ES doesn't support array constructors.
        assignments = ""
        for i, v in enumerate(m.flatten()):
            assignments += f"    bayer[{i}] = {v:.6f};\n"
        return f"""{GLSL_HEADER}
void main() {{
    {_glsl_aspect_correction()}
    float bayer[{n*n}];
{assignments}    int x = int(mod(frag.x, {float(n):.1f}));
    int y = int(mod(frag.y, {float(n):.1f}));
    float t = bayer[y * {n} + x];
    vec4 src = texture2D(u_texture, uv_tex);
    float lum = clamp({_glsl_lum()} + ({bias:.4f}), 0.0, 1.0);
    float coverage = step(lum, t);
    gl_FragColor = vec4(mix(u_paper, u_ink, coverage), 1.0);
}}
"""

    if mode == "Erwin grain (reticulated)":
        seed = float(p.get("seed", 2))
        return f"""{GLSL_HEADER}
// Hash + smoothstep value noise — the GPU equivalent of the CPU path's
// gaussian-filtered random field. Cell size scales with grain_px so the
// noise's feature size matches the sin/cos pattern's period.
float hash21(vec2 p) {{
    p = fract(p * vec2(443.8975, 397.2973));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}}
float vnoise(vec2 p) {{
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}}
void main() {{
    {_glsl_aspect_correction()}
    float grain = {p['grain']:.4f};
    float retic = {p['retic']:.4f};
    float seed = {seed:.1f};
    vec2 nc = frag / max(grain, 1.0);
    // Two decorrelated noise samples; subtract 0.5 → zero-mean, scale by
    // sqrt(12) so std≈1 (matches the CPU normalisation).
    float n1 = (vnoise(nc + vec2(seed * 1.3, seed * 2.7)) - 0.5) * 3.4641;
    float n2 = (vnoise(nc + vec2(seed * 5.1 + 31.7, seed * 4.3 + 11.3)) - 0.5) * 3.4641;
    float warp = retic * grain * 3.0;
    float sx = frag.x + warp * n1;
    float sy = frag.y + warp * n2;
    float period = max(1.0, grain * 1.4);
    float base = sin(6.2831853 * sx / period) * cos(6.2831853 * sy / period);
    float pat = 0.5 + 0.5 * base;
    vec4 src = texture2D(u_texture, uv_tex);
    float lum = {_glsl_lum()};
    float coverage = step(lum, pat);
    gl_FragColor = vec4(mix(u_paper, u_ink, coverage), 1.0);
}}
"""

    return f"// Shader generation not supported for: {mode}\n"


def generate_metal(mode: str, p: dict, ink: np.ndarray,
                   paper: np.ndarray) -> str:
    """Return a Metal Shading Language fragment shader. Same logic as the
    GLSL generator but in MSL syntax. Modes not yet ported fall back to a
    stub comment."""
    if mode == "Ives/Levy (AM dot)":
        halo_block = ""
        if p.get("halo", 0.0) > 0.0:
            halo_block = f"""\
    float ring = step(d, target + 0.18) * (1.0 - step(d, target * 0.55));
    float ctr  = step(d, target * 0.45);
    coverage = saturate(coverage + {p['halo']:.4f} * ring
                                 - 0.5 * {p['halo']:.4f} * ctr);
"""
        return f"""{METAL_HEADER}
fragment float4 halftone_am(
    VOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    sampler samp [[sampler(0)]],
    constant float2& resolution [[buffer(0)]],
    constant float3& inkColor [[buffer(1)]],
    constant float3& paperColor [[buffer(2)]])
{{
    float2 frag = in.uv * resolution;
    float theta = {p['angle']:.4f} * 3.14159265 / 180.0;
    float c = cos(theta), s = sin(theta);
    float2 rot = float2(c * frag.x + s * frag.y, -s * frag.x + c * frag.y);
    float cell_px = {max(1.0, p['dpi'] / max(1, p['lpi'])):.4f};
    float2 cf = fract(rot / cell_px);
    float4 src = tex.sample(samp, in.uv);
    float lum = dot(src.rgb, float3(0.2126, 0.7152, 0.0722));
    {('lum = 1.0 - lum;' if p.get('invert') else '')}
    lum = saturate(lum);
    lum = pow(lum, {1.0 / max(0.05, p.get('tone', 1.0)):.4f});
    float target = clamp(sqrt(1.0 - lum), {p['dot_range'][0]:.4f}, {p['dot_range'][1]:.4f});
    float du = cf.x - 0.5, dv = cf.y - 0.5;
    float d = {_shape_metal(p['shape'])};
    float coverage = step(d, target);
{halo_block}
    return float4(mix(paperColor, inkColor, coverage), 1.0);
}}
"""

    if mode == "Line screen (Akrography)":
        cell = max(1.0, p['dpi'] / max(1, p['lpi']))
        cross = ""
        if p.get("crosshatch"):
            cross = f"""\
    {{
        float theta2 = ({p['angle'] + 90.0:.4f}) * 3.14159265 / 180.0;
        float c2 = cos(theta2), s2 = sin(theta2);
        float v2 = -s2 * frag.x + c2 * frag.y;
        float vf2 = fract(v2 / {cell:.4f});
        float t2 = target * 0.6;
        float c2_cov = step(fabs(vf2 - 0.5), t2 * 0.5);
        coverage = saturate(coverage + c2_cov - coverage * c2_cov);
    }}
"""
        return f"""{METAL_HEADER}
fragment float4 halftone_line(
    VOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    sampler samp [[sampler(0)]],
    constant float2& resolution [[buffer(0)]],
    constant float3& inkColor [[buffer(1)]],
    constant float3& paperColor [[buffer(2)]])
{{
    float2 frag = in.uv * resolution;
    float theta = {p['angle']:.4f} * 3.14159265 / 180.0;
    float c = cos(theta), s = sin(theta);
    float v = -s * frag.x + c * frag.y;
    float vf = fract(v / {cell:.4f});
    float4 src = tex.sample(samp, in.uv);
    float lum = dot(src.rgb, float3(0.2126, 0.7152, 0.0722));
    float target = 1.0 - saturate(lum);
    float coverage = step(fabs(vf - 0.5), target * 0.5);
{cross}
    return float4(mix(paperColor, inkColor, coverage), 1.0);
}}
"""

    if mode == "Bayer ordered dither":
        n = int(p["n"])
        m = bayer_matrix(n)
        m_str = ", ".join(f"{v:.6f}h" for v in m.flatten())
        bias = p.get("thresh_bias", 0.0)
        return f"""{METAL_HEADER}
constant half bayer[{n*n}] = {{ {m_str} }};

fragment float4 halftone_bayer(
    VOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    sampler samp [[sampler(0)]],
    constant float2& resolution [[buffer(0)]],
    constant float3& inkColor [[buffer(1)]],
    constant float3& paperColor [[buffer(2)]])
{{
    float2 frag = in.uv * resolution;
    int x = int(fmod(frag.x, {float(n):.1f}));
    int y = int(fmod(frag.y, {float(n):.1f}));
    float t = float(bayer[y * {n} + x]);
    float4 src = tex.sample(samp, in.uv);
    float lum = saturate(dot(src.rgb, float3(0.2126, 0.7152, 0.0722))
                          + {bias:.4f});
    float coverage = step(lum, t);
    return float4(mix(paperColor, inkColor, coverage), 1.0);
}}
"""

    if mode == "Wavy-line screen":
        cell = max(1.0, p['dpi'] / max(1, p['lpi']))
        return f"""{METAL_HEADER}
fragment float4 halftone_wavy(
    VOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    sampler samp [[sampler(0)]],
    constant float2& resolution [[buffer(0)]],
    constant float3& inkColor [[buffer(1)]],
    constant float3& paperColor [[buffer(2)]])
{{
    float2 frag = in.uv * resolution;
    float theta = {p['angle']:.4f} * 3.14159265 / 180.0;
    float c = cos(theta), s = sin(theta);
    float u = c * frag.x + s * frag.y;
    float v = -s * frag.x + c * frag.y;
    float u_n = u / {cell:.4f};
    float phase = sin(2.0 * 3.14159265 * {p['freq']:.4f} * u_n) * {p['amp']:.4f};
    float vf = fract(v / {cell:.4f} + phase);
    float4 src = tex.sample(samp, in.uv);
    float lum = dot(src.rgb, float3(0.2126, 0.7152, 0.0722));
    float target = 1.0 - saturate(lum);
    float coverage = step(fabs(vf - 0.5), target * 0.5);
    return float4(mix(paperColor, inkColor, coverage), 1.0);
}}
"""

    if mode == "Stochastic / FM (Autotype)":
        return f"""{METAL_HEADER}
inline float hash21(float2 p) {{
    p = fract(p * float2(443.8975, 397.2973));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}}

fragment float4 halftone_fm(
    VOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    sampler samp [[sampler(0)]],
    constant float2& resolution [[buffer(0)]],
    constant float3& inkColor [[buffer(1)]],
    constant float3& paperColor [[buffer(2)]])
{{
    float2 frag = in.uv * resolution;
    float grain = {p['grain']:.4f};
    float2 cell_xy = floor(frag / grain);
    float noise_lo = hash21(cell_xy);
    float noise_hi = hash21(frag);
    float noise = mix(noise_lo, noise_hi, {p['hi_freq']:.4f});
    float4 src = tex.sample(samp, in.uv);
    float lum = dot(src.rgb, float3(0.2126, 0.7152, 0.0722));
    float coverage = step(lum, noise);
    return float4(mix(paperColor, inkColor, coverage), 1.0);
}}
"""

    if mode == "Erwin grain (reticulated)":
        seed = float(p.get("seed", 2))
        return f"""{METAL_HEADER}
inline float hash21(float2 p) {{
    p = fract(p * float2(443.8975, 397.2973));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}}
inline float vnoise(float2 p) {{
    float2 i = floor(p), f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + float2(1.0, 0.0));
    float c = hash21(i + float2(0.0, 1.0));
    float d = hash21(i + float2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}}

fragment float4 halftone_erwin(
    VOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    sampler samp [[sampler(0)]],
    constant float2& resolution [[buffer(0)]],
    constant float3& inkColor [[buffer(1)]],
    constant float3& paperColor [[buffer(2)]])
{{
    float2 frag = in.uv * resolution;
    float grain = {p['grain']:.4f};
    float retic = {p['retic']:.4f};
    float seed = {seed:.1f};
    float2 nc = frag / max(grain, 1.0);
    float n1 = (vnoise(nc + float2(seed * 1.3, seed * 2.7)) - 0.5) * 3.4641;
    float n2 = (vnoise(nc + float2(seed * 5.1 + 31.7, seed * 4.3 + 11.3)) - 0.5) * 3.4641;
    float warp = retic * grain * 3.0;
    float sx = frag.x + warp * n1;
    float sy = frag.y + warp * n2;
    float period = max(1.0, grain * 1.4);
    float base = sin(6.2831853 * sx / period) * cos(6.2831853 * sy / period);
    float pat = 0.5 + 0.5 * base;
    float4 src = tex.sample(samp, in.uv);
    float lum = dot(src.rgb, float3(0.2126, 0.7152, 0.0722));
    float coverage = step(lum, pat);
    return float4(mix(paperColor, inkColor, coverage), 1.0);
}}
"""

    if mode == "CMYK process color":
        ang = {"C": p["ang_c"], "M": p["ang_m"],
               "Y": p["ang_y"], "K": p["ang_k"]}
        cell = max(1.0, p['dpi'] / max(1, p['lpi']))
        shape_expr = _shape_metal(p['shape'])
        ink_const = {
            "C": "float3(0.0, 0.68, 0.94)",
            "M": "float3(0.93, 0.0, 0.55)",
            "Y": "float3(0.98, 0.93, 0.0)",
            "K": "float3(0.05, 0.05, 0.05)",
        }
        chans = p.get("channels") or ["C", "M", "Y", "K"]
        block = ""
        for ch in chans:
            block += f"""\
    {{
        float theta = ({ang[ch]:.4f}) * 3.14159265 / 180.0;
        float c = cos(theta), s = sin(theta);
        float2 rot = float2(c * frag.x + s * frag.y, -s * frag.x + c * frag.y);
        float2 cf = fract(rot / {cell:.4f});
        float target = sqrt(1.0 - saturate(sep_{ch.lower()}));
        float du = cf.x - 0.5, dv = cf.y - 0.5;
        float d = {shape_expr};
        float cov = step(d, target);
        out_rgb = out_rgb * mix(float3(1.0), {ink_const[ch]}, cov);
    }}
"""
        return f"""{METAL_HEADER}
fragment float4 halftone_cmyk(
    VOut in [[stage_in]],
    texture2d<float> tex [[texture(0)]],
    sampler samp [[sampler(0)]],
    constant float2& resolution [[buffer(0)]],
    constant float3& inkColor [[buffer(1)]],
    constant float3& paperColor [[buffer(2)]])
{{
    float2 frag = in.uv * resolution;
    float4 src = tex.sample(samp, in.uv);
    float k = 1.0 - max(max(src.r, src.g), src.b);
    float denom = (k < 1.0) ? (1.0 - k) : 1.0;
    float sep_c = saturate((1.0 - src.r - k) / denom);
    float sep_m = saturate((1.0 - src.g - k) / denom);
    float sep_y = saturate((1.0 - src.b - k) / denom);
    float sep_k = saturate(k);
    float3 out_rgb = paperColor;
{block}
    return float4(out_rgb, 1.0);
}}
"""

    return f"// Shader generation not supported for: {mode}\n"


def webgl_preview_html(glsl_frag: str, image_data_url: str,
                       ink_hex: str, paper_hex: str,
                       img_w: int, img_h: int,
                       canvas_h: int = 460) -> str:
    """Return a self-contained HTML page that compiles and runs the GLSL
    fragment shader on the embedded source image. Preserves aspect ratio."""
    def _hex_to_floats(h: str) -> str:
        h = h.lstrip("#")
        return ", ".join(f"{int(h[i:i+2], 16) / 255:.4f}"
                         for i in (0, 2, 4))

    # Escape backticks / </script> so the shader can sit inside a JS string.
    js_safe = glsl_frag.replace("\\", "\\\\").replace("`", "\\`")
    return f"""
<!DOCTYPE html>
<html><head><style>
body {{ margin: 0; background: #1e1e1e; color: #ddd; font-family: -apple-system, sans-serif; }}
canvas {{ display: block; width: 100%; height: {canvas_h}px;
          background: #000; image-rendering: pixelated; }}
#log {{ padding: 4px 8px; font: 11px/1.3 ui-monospace, monospace; color: #f88; min-height: 14px; }}
</style></head><body>
<canvas id="c"></canvas>
<div id="log"></div>
<script>
(function() {{
  const fragSrc = `{js_safe}`;
  const vertSrc = `
    attribute vec2 a_pos;
    varying vec2 v_uv;
    void main() {{ v_uv = a_pos * 0.5 + 0.5;
                   v_uv.y = 1.0 - v_uv.y;
                   gl_Position = vec4(a_pos, 0.0, 1.0); }}`;
  const canvas = document.getElementById('c');
  const log = document.getElementById('log');
  const gl = canvas.getContext('webgl', {{ preserveDrawingBuffer: true }});
  if (!gl) {{ log.textContent = 'WebGL unavailable.'; return; }}

  function compile(type, src) {{
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {{
      log.textContent = 'shader error: ' + gl.getShaderInfoLog(sh);
      return null;
    }}
    return sh;
  }}
  const vs = compile(gl.VERTEX_SHADER, vertSrc);
  const fs = compile(gl.FRAGMENT_SHADER, fragSrc);
  if (!vs || !fs) return;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs); gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {{
    log.textContent = 'link error: ' + gl.getProgramInfoLog(prog);
    return;
  }}
  gl.useProgram(prog);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER,
                new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]),
                gl.STATIC_DRAW);
  const posLoc = gl.getAttribLocation(prog, 'a_pos');
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  // 1x1 placeholder until the image arrives.
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA,
                gl.UNSIGNED_BYTE, new Uint8Array([255,255,255,255]));

  const uTex = gl.getUniformLocation(prog, 'u_texture');
  const uRes = gl.getUniformLocation(prog, 'u_resolution');
  const uInk = gl.getUniformLocation(prog, 'u_ink');
  const uPap = gl.getUniformLocation(prog, 'u_paper');
  const uImgSize = gl.getUniformLocation(prog, 'u_image_size');
  gl.uniform1i(uTex, 0);
  gl.uniform3f(uInk, {_hex_to_floats(ink_hex)});
  gl.uniform3f(uPap, {_hex_to_floats(paper_hex)});
  gl.uniform2f(uImgSize, {img_w}.0, {img_h}.0);

  function render() {{
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w; canvas.height = h;
    gl.viewport(0, 0, w, h);
    gl.uniform2f(uRes, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }}

  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = function() {{
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA,
                  gl.UNSIGNED_BYTE, img);
    render();
  }};
  img.onerror = function() {{ log.textContent = 'image load failed'; render(); }};
  img.src = "{image_data_url}";

  window.addEventListener('resize', render);
}})();
</script>
</body></html>"""


def _image_to_data_url_cached(img_rgb: np.ndarray, max_dim: int = 800) -> str:
    """Encode an RGB numpy image (values in [0,1]) as a base64 data URL,
    downsampling so the inline payload stays small. Internal cached version."""
    arr = (np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr)
    if max(pil.size) > max_dim:
        ratio = max_dim / max(pil.size)
        pil = pil.resize((int(pil.width * ratio), int(pil.height * ratio)),
                         Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

@st.cache_data(show_spinner=False)
def image_to_data_url(img_rgb: np.ndarray, max_dim: int = 800) -> str:
    """Cached wrapper for expensive PNG encoding."""
    return _image_to_data_url_cached(img_rgb, max_dim)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

# Each mode gets a short label + a single-glyph icon. The full name remains
# available in tooltips / the about expander.
MODE_META = [
    ("Ives/Levy (AM dot)",            "● AM"),
    ("CMYK process color",            "◐ CMYK"),
    ("Line screen (Akrography)",      "│ Line"),
    ("Wavy-line screen",              "∿ Wavy"),
    ("Stochastic / FM (Autotype)",    "⁂ FM"),
    ("Metzograph (cracked)",          "⌇ Metz"),
    ("Erwin grain (reticulated)",     "⌬ Erwin"),
    ("Bayer ordered dither",          "▦ Bayer"),
    ("Floyd–Steinberg error diffusion", "▨ FS"),
]
MODES = [m for m, _ in MODE_META]
MODE_LABELS = [lbl for _, lbl in MODE_META]
LABEL_TO_MODE = dict(zip(MODE_LABELS, MODES))

SRC_META = [
    ("Apple still life",   "🍎"),
    ("Shaded sphere",      "◯"),
    ("Color test",         "◐"),
    ("Linear gradient",    "▭"),
    ("Radial gradient",    "◉"),
    ("Step wedge",         "▤"),
    ("Upload...",          "⇪"),
]
SRC_LABELS = [lbl for _, lbl in SRC_META]
LABEL_TO_SRC = {lbl: name for (name, _), lbl in zip(SRC_META, SRC_LABELS)}

with st.sidebar:
    mode_label = st.segmented_control(
        "screen", MODE_LABELS, default=MODE_LABELS[0],
        label_visibility="collapsed")
    mode = LABEL_TO_MODE[mode_label or MODE_LABELS[0]]

    src_label = st.segmented_control(
        "source", SRC_LABELS, default=SRC_LABELS[0],
        label_visibility="collapsed")
    src_kind = LABEL_TO_SRC[src_label or SRC_LABELS[0]]

    uploaded = None
    if src_kind == "Upload...":
        uploaded = st.file_uploader("img", type=["png", "jpg", "jpeg",
                                                  "bmp", "tif"],
                                    label_visibility="collapsed")

    # Inks on their own row (color pickers don't need much width).
    c_ink, c_paper = st.columns(2)
    with c_ink:
        ink_hex = st.color_picker("✒ ink", "#111111")
    with c_paper:
        paper_hex = st.color_picker("▢ paper", "#fefdf6")

    # Resolution gets the full sidebar width so all tick labels are legible.
    size = st.select_slider(
        "resolution (px)",
        options=[192, 256, 320, 384, 448, 512, 640, 768, 896,
                 1024, 1280, 1536, 2048],
        value=1024)

    # --- Global pre-processing & paper -----------------------------------
    with st.expander("✦ image", expanded=False):
        # Dot gain section — tone transfer curves to compensate for ink spread
        st.caption("Dot gain (ink spread on paper)")

        # Paper type presets for convenience
        preset_gains = {
            "none": 0.0,
            "newsprint (15%)": 15.0,
            "coated (10%)": 10.0,
            "uncoated (22%)": 22.0,
            "offset (12%)": 12.0,
        }
        dot_gain_preset = st.selectbox(
            "paper type",
            list(preset_gains.keys()),
            index=0, label_visibility="collapsed")

        # Slider for manual control
        dot_gain_percent = st.slider(
            "dot gain %",
            0.0, 50.0, preset_gains[dot_gain_preset], step=0.5,
            help="Percentage ink spreads beyond halftone dot. Typical: newsprint 10-20%, coated 5-15%, uncoated 15-30%")

        # Convert dot gain % to tone gamma for compensation
        # Higher dot gain requires more tonal darkening (lower gamma) to compensate
        if dot_gain_percent > 0:
            dot_gain_gamma = max(0.3, 1.0 - (dot_gain_percent / 100.0) * 0.4)
        else:
            dot_gain_gamma = 1.0

        pre_gamma = st.slider("gamma", 0.2, 3.0, dot_gain_gamma, step=0.01)
        pre_bright = st.slider("brightness", -0.5, 0.5, 0.0, step=0.005)
        pre_contrast = st.slider("contrast", 0.2, 3.0, 1.0, step=0.01)
        pre_blur = st.slider("blur σ", 0.0, 8.0, 0.0, step=0.05)
        pre_sharp = st.slider("sharpen", 0.0, 3.0, 0.0, step=0.02)
        pre_sat = st.slider("saturation", 0.0, 2.5, 1.0, step=0.01)
        pre_post = st.slider("posterize (tone steps)", 0, 32, 0, step=1)
        pre_vignette = st.slider("vignette", 0.0, 1.0, 0.0, step=0.01)
        pre_threshold = st.slider("threshold (binarise)",
                                  0.0, 1.0, 0.0, step=0.005)
        lvl_black, lvl_white = st.slider(
            "levels: black / white",
            0.0, 1.0, (0.0, 1.0), step=0.005)
        lvl_mid = st.slider("levels: mid (γ pivot)",
                            0.05, 0.95, 0.5, step=0.005)
        pre_invert_src = st.checkbox("invert source")

    with st.expander("◳ paper", expanded=False):
        paper_tex_kind = st.selectbox("texture", PAPER_KINDS)
        paper_tex_strength = st.slider("strength", 0.0, 1.0, 0.0, step=0.01)

    # Mode-specific parameters — packed two-per-row where possible.
    params: dict = {}
    if mode == "Ives/Levy (AM dot)":
        params["lpi"] = st.slider("lpi (screen ruling)", 5, 400, 60, step=1)
        params["dpi"] = st.slider("dpi (output res)", 100, 2400, 400, step=10)
        params["angle"] = st.slider("screen angle °",
                                    0.0, 90.0, 45.0, step=0.1)
        params["shape"] = st.selectbox(
            "dot shape",
            ["round", "square", "diamond", "elliptical", "line"])
        params["halo"] = st.slider("letterpress halo",
                                   0.0, 1.0, 0.0, step=0.01)
        params["jitter"] = st.slider("positional jitter",
                                     0.0, 0.5, 0.0, step=0.005)

        # Tone transfer curve section for advanced dot gain compensation
        st.caption("Tone transfer curve (dot gain compensation)")
        tone_curve_mode = st.radio(
            "curve type", ["auto", "shadows", "midtones", "highlights"],
            horizontal=True, label_visibility="collapsed")

        # Apply different curves based on where dot gain is most visible
        tone_defaults = {
            "auto": 1.0,        # Neutral gamma (global adjustment)
            "shadows": 1.2,     # Lift shadows (dot gain most visible in shadows)
            "midtones": 0.85,   # Darken midtones (dot gain causes loss of detail)
            "highlights": 1.1,  # Lift highlights slightly
        }
        params["tone"] = st.slider("tone γ (dot gain correction)",
                                   0.2, 3.0, tone_defaults[tone_curve_mode], step=0.01,
                                   help="<1.0 darkens (compensate spreading), >1.0 lightens")

        params["dot_range"] = st.slider("printable dot range",
                                        0.0, 1.2, (0.0, 1.0), step=0.01)
        params["invert"] = st.checkbox("negative")

    elif mode == "CMYK process color":
        params["lpi"] = st.slider("lpi (screen ruling)", 10, 400, 85, step=1)
        params["dpi"] = st.slider("dpi (output res)", 100, 2400, 500, step=10)
        params["shape"] = st.selectbox(
            "dot shape", ["round", "elliptical", "square", "diamond"])
        params["halo"] = st.slider("letterpress halo",
                                   0.0, 1.0, 0.0, step=0.01)

        # Dot gain compensation for CMYK
        st.caption("Dot gain per ink")
        params["cmyk_gain_c"] = st.slider("cyan dot gain %", 0.0, 30.0, 0.0, step=0.5)
        params["cmyk_gain_m"] = st.slider("magenta dot gain %", 0.0, 30.0, 0.0, step=0.5)
        params["cmyk_gain_y"] = st.slider("yellow dot gain %", 0.0, 30.0, 0.0, step=0.5)
        params["cmyk_gain_k"] = st.slider("black dot gain %", 0.0, 30.0, 0.0, step=0.5)

        params["density"] = st.slider("ink density",
                                      0.2, 2.0, 1.0, step=0.01)
        params["k_strength"] = st.slider("K boost (detail in black)",
                                         0.0, 2.5, 1.0, step=0.01)
        st.caption("screen angles")
        params["ang_c"] = st.slider("cyan °",    0.0, 180.0,
                                    CLASSIC_ANGLES["C"], step=0.1)
        params["ang_m"] = st.slider("magenta °", 0.0, 180.0,
                                    CLASSIC_ANGLES["M"], step=0.1)
        params["ang_y"] = st.slider("yellow °",  0.0, 180.0,
                                    CLASSIC_ANGLES["Y"], step=0.1)
        params["ang_k"] = st.slider("black °",   0.0, 180.0,
                                    CLASSIC_ANGLES["K"], step=0.1)
        params["channels"] = st.segmented_control(
            "channels", ["C", "M", "Y", "K"],
            default=["C", "M", "Y", "K"], selection_mode="multi")

    elif mode == "Line screen (Akrography)":
        params["lpi"] = st.slider("lpi (lines per inch)",
                                  5, 400, 80, step=1)
        params["dpi"] = st.slider("dpi (output res)",
                                  100, 2400, 400, step=10)
        params["angle"] = st.slider("line angle °",
                                    0.0, 180.0, 0.0, step=0.1)
        params["soft"] = st.slider("soft edge (anti-alias)",
                                   0.0, 0.3, 0.0, step=0.002)
        params["crosshatch"] = st.checkbox("crosshatch (second perpendicular set)",
                                            value=False)

    elif mode == "Wavy-line screen":
        params["lpi"] = st.slider("lpi (lines per inch)",
                                  5, 400, 60, step=1)
        params["dpi"] = st.slider("dpi (output res)",
                                  100, 2400, 400, step=10)
        params["angle"] = st.slider("line angle °",
                                    0.0, 180.0, 0.0, step=0.1)
        params["amp"] = st.slider("wave amplitude (cells)",
                                  0.0, 1.5, 0.25, step=0.005)
        params["freq"] = st.slider("wave frequency (cycles/cell)",
                                   0.05, 8.0, 0.6, step=0.01)

    elif mode == "Stochastic / FM (Autotype)":
        params["grain"] = st.slider("grain size (px)",
                                    0.5, 16.0, 3.0, step=0.05)
        params["hi_freq"] = st.slider("high-frequency dither mix",
                                      0.0, 1.0, 0.3, step=0.01)
        params["seed"] = max(0, int(st.number_input("random seed", value=0, step=1)))

    elif mode == "Metzograph (cracked)":
        params["cell"] = st.slider("pattern scale (px)",
                                   0.5, 20.0, 4.0, step=0.05)
        params["cracks"] = st.slider("crack density",
                                     0.0, 1.0, 0.4, step=0.01)
        params["seed"] = max(0, int(st.number_input("random seed", value=1, step=1)))

    elif mode == "Erwin grain (reticulated)":
        params["grain"] = st.slider("grain scale (px)",
                                    1.0, 24.0, 6.0, step=0.05)
        params["retic"] = st.slider("reticulation (warp strength)",
                                    0.0, 2.5, 0.6, step=0.01)
        params["seed"] = max(0, int(st.number_input("random seed", value=2, step=1)))

    elif mode == "Bayer ordered dither":
        params["n"] = st.select_slider(
            "matrix size", options=[2, 4, 8, 16, 32, 64], value=8)
        params["thresh_bias"] = st.slider(
            "threshold bias (-light / +dark)",
            -0.4, 0.4, 0.0, step=0.005)

    elif mode == "Floyd–Steinberg error diffusion":
        params["kernel"] = st.selectbox(
            "error-diffusion kernel",
            ["Floyd–Steinberg", "Atkinson", "Jarvis-J-N", "Stucki",
             "Sierra-3", "Burkes"])
        params["serpentine"] = st.checkbox("serpentine traversal",
                                            value=True)
        if params["kernel"] != "Floyd–Steinberg":
            st.caption("non-FS kernels run pure-Python; small sizes only.")

    show_source = st.toggle("show source", value=False)

# --- Build source image --------------------------------------------------

if src_kind == "Linear gradient":
    src_gray = make_linear_gradient(size)
    src_rgb = np.stack([src_gray] * 3, axis=-1)
elif src_kind == "Radial gradient":
    src_gray = make_radial_gradient(size)
    src_rgb = np.stack([src_gray] * 3, axis=-1)
elif src_kind == "Step wedge":
    src_gray = make_step_wedge(size)
    src_rgb = np.stack([src_gray] * 3, axis=-1)
elif src_kind == "Apple still life":
    src_rgb = make_apple_still_life(size)
    src_gray = to_grayscale(src_rgb)
elif src_kind == "Shaded sphere":
    src_gray = make_shaded_sphere(size)
    src_rgb = np.stack([src_gray] * 3, axis=-1)
elif src_kind == "Color test":
    src_rgb = make_color_test(size)
    src_gray = to_grayscale(src_rgb)
else:
    if uploaded is None:
        st.info("Upload an image to continue, or pick a built-in pattern.")
        st.stop()
    src_rgb = load_user_image(uploaded, size)
    src_gray = to_grayscale(src_rgb)
    size = max(src_rgb.shape[:2])

ink = hex_to_rgb01(ink_hex)
paper = hex_to_rgb01(paper_hex)

# Apply global pre-processing to both grayscale and RGB sources.
if pre_invert_src:
    src_gray = 1.0 - src_gray
    src_rgb = 1.0 - src_rgb
_pre_kwargs = dict(
    gamma=pre_gamma, brightness=pre_bright, contrast=pre_contrast,
    blur_sigma=pre_blur, sharpen=pre_sharp,
    black_point=lvl_black, white_point=lvl_white, mid_point=lvl_mid,
    saturation=pre_sat, posterize=int(pre_post),
    vignette=pre_vignette, threshold=pre_threshold,
)
src_gray = preprocess(src_gray, **_pre_kwargs)
src_rgb = preprocess(src_rgb, **_pre_kwargs)

# --- Render --------------------------------------------------------------

if mode == "Ives/Levy (AM dot)":
    dot_lo, dot_hi = params["dot_range"]
    img = am_halftone(src_gray,
                      AMParams(lpi=params["lpi"], dpi=params["dpi"],
                               angle_deg=params["angle"],
                               shape=params["shape"], halo=params["halo"],
                               jitter=params["jitter"],
                               invert=params["invert"],
                               tone_gamma=params["tone"],
                               min_dot=dot_lo, max_dot=dot_hi),
                      ink, paper)
elif mode == "CMYK process color":
    # K-boost: bias the K plane by exponent before screening so more or less
    # detail is carried by black.
    rgb_for_cmyk = src_rgb.copy()
    if params["k_strength"] != 1.0:
        lum = to_grayscale(rgb_for_cmyk)
        rgb_for_cmyk = rgb_for_cmyk * (lum ** (1 - params["k_strength"]))[..., None]
        rgb_for_cmyk = np.clip(rgb_for_cmyk, 0.0, 1.0)

    # Per-channel dot gain compensation
    dot_gains = {
        "C": params.get("cmyk_gain_c", 0.0),
        "M": params.get("cmyk_gain_m", 0.0),
        "Y": params.get("cmyk_gain_y", 0.0),
        "K": params.get("cmyk_gain_k", 0.0),
    }

    img = cmyk_halftone(
        rgb_for_cmyk,
        AMParams(lpi=params["lpi"], dpi=params["dpi"],
                 shape=params["shape"], halo=params["halo"]),
        angles={"C": params["ang_c"], "M": params["ang_m"],
                "Y": params["ang_y"], "K": params["ang_k"]},
        paper_rgb=paper,
        channels=tuple(params["channels"] or ()) or ("K",),
        dot_gains=dot_gains)
    # Apply ink density by blending toward paper white.
    if params["density"] != 1.0:
        d = params["density"]
        img = paper * (1.0 - d) + img * d if d < 1.0 else \
              np.clip(img ** (1.0 / d), 0.0, 1.0)
elif mode == "Line screen (Akrography)":
    img = line_screen(src_gray, params["lpi"], params["dpi"],
                      params["angle"], ink, paper,
                      crosshatch=params["crosshatch"],
                      soft_edge=params["soft"])
elif mode == "Wavy-line screen":
    img = wavy_line_screen(src_gray, params["lpi"], params["dpi"],
                           params["angle"], params["amp"], params["freq"],
                           ink, paper)
elif mode == "Stochastic / FM (Autotype)":
    img = stochastic_screen(src_gray, params["grain"], ink, paper,
                            seed=params["seed"], hi_freq=params["hi_freq"])
elif mode == "Metzograph (cracked)":
    img = metzograph_screen(src_gray, params["cell"], params["cracks"],
                            ink, paper, seed=params["seed"])
elif mode == "Erwin grain (reticulated)":
    img = erwin_grain_screen(src_gray, params["grain"], params["retic"],
                             ink, paper, seed=params["seed"])
elif mode == "Bayer ordered dither":
    biased = np.clip(src_gray + params["thresh_bias"], 0.0, 1.0)
    img = bayer_dither(biased, params["n"], ink, paper)
elif mode == "Floyd–Steinberg error diffusion":
    if params["kernel"] == "Floyd–Steinberg" and params["serpentine"]:
        # PIL's fast C path — only valid for the default kernel.
        img = floyd_steinberg(src_gray, ink, paper)
    else:
        img = error_diffuse(src_gray, params["kernel"],
                            params["serpentine"], ink, paper)
else:
    st.error(f"Unknown mode: {mode}")
    st.stop()

# Apply paper texture overlay (multiplicative).
if paper_tex_kind != "none" and paper_tex_strength > 0.0:
    tex = paper_texture(paper_tex_kind, img.shape[:2], paper_tex_strength)
    img = apply_paper(img, tex)

# --- Display -------------------------------------------------------------

result_clipped = np.clip(img, 0.0, 1.0)

# Two-tab layout: preview vs. shader. Keeps the shader UI out of the way of
# the halftone image when the user is just exploring.
tab_preview, tab_shader = st.tabs(["🖼  Halftone preview",
                                    "⚡ GPU shader"])

with tab_preview:
    if show_source:
        c_src, c_out = st.columns([1, 3])
        with c_src:
            st.image(src_rgb, caption="source",
                     use_container_width=True, clamp=True)
        with c_out:
            st.image(result_clipped, caption=mode,
                     use_container_width=True, clamp=True)
    else:
        st.image(result_clipped, use_container_width=True, clamp=True)

    c_dl, c_about = st.columns([1, 4])
    with c_dl:
        buf = io.BytesIO()
        Image.fromarray((result_clipped * 255.0).astype(np.uint8)).save(
            buf, format="PNG")
        st.download_button(
            "Download PNG", data=buf.getvalue(),
            file_name=f"halftone_{mode.split()[0].lower()}.png",
            mime="image/png", use_container_width=True)
    with c_about:
        with st.expander("About the screens"):
            st.markdown("""
**Ives/Levy AM dot** — canonical halftone screen. Dot *size* varies with
tone, dot *spacing* is constant. Default newspaper angle is 45°.

**CMYK process color** — four AM screens, one per ink, rotated to different
angles (classically C 15°, M 75°, Y 0°, K 45°) so the grids interleave
instead of forming moiré.

**Line screen (Akrography)** — parallel rules whose thickness varies with
tone. Cheap, used for tinted backgrounds and 'Akrotone' prints.

**Wavy-line** — line screen with a sinusoidal modulation; one of the
irregular screens identifiable under magnification.

**Stochastic / FM (Autotype grain)** — randomly placed *fixed-size* dots
whose *density* varies with tone instead of dot size. Avoids moiré.

**Metzograph** — cracked-resist screen approximated by thresholding
low-pass-filtered noise with an added thin 'crack' component.

**Erwin grain** — reticulated gelatin screen (1926); long sinuous filaments
rather than discrete dots. Implemented as a warped grid.

**Bayer ordered dither** — modern, deterministic threshold-map dithering
using a recursive 2ⁿ×2ⁿ Bayer matrix.

**Floyd–Steinberg** — error-diffusion algorithm (1976); each pixel pushes
its quantization error onto its four unprocessed neighbours.
""")

with tab_shader:
    if mode not in SHADER_MODES:
        st.warning(
            f"**{mode}** doesn't map to a stateless per-pixel shader "
            "(error diffusion is sequential; Metzograph needs a crack "
            "texture). Pick AM, CMYK, Line, Wavy, FM, Bayer, or Erwin in "
            "the sidebar to generate a shader."
        )
    else:
        glsl_src = generate_glsl(mode, params, ink, paper)
        metal_src = generate_metal(mode, params, ink, paper)
        slug = mode.split()[0].lower().rstrip(",.()")

        st.markdown("**Live WebGL preview** — your current image, "
                    "screened on the GPU by the generated GLSL shader.")
        h, w = src_rgb.shape[:2]
        data_url = image_to_data_url(src_rgb)
        components.html(
            webgl_preview_html(glsl_src, data_url, ink_hex, paper_hex,
                               w, h, canvas_h=460),
            height=500)

        c_g, c_m = st.columns(2)
        with c_g:
            st.download_button(
                "Download GLSL (.frag)", data=glsl_src,
                file_name=f"halftone_{slug}.frag", mime="text/plain",
                use_container_width=True)
        with c_m:
            st.download_button(
                "Download Metal (.metal)", data=metal_src,
                file_name=f"halftone_{slug}.metal", mime="text/plain",
                use_container_width=True)

        tab_glsl, tab_metal = st.tabs(["GLSL fragment", "Metal fragment"])
        with tab_glsl:
            st.code(glsl_src, language="glsl")
        with tab_metal:
            st.code(metal_src, language="cpp")
