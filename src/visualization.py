"""
Minimal utilities for drawing relevancy heatmap overlays.

The defaults are tuned to produce a "focused" look rather than the noisy
confetti pattern you get from raw `llama` rollout with flat 50% alpha:

- Percentile normalization suppresses the low-end noise floor.
- Value-dependent alpha keeps low-relevance areas transparent so the
  underlying image stays readable.
- A small Gaussian blur smooths over patch-grid artifacts (24x24 -> HxW).
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

cmap = plt.get_cmap("jet")


def _normalize(mat: np.ndarray, p_low: float = 60.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile-based min-max normalization to kill the noise floor."""
    lo = np.percentile(mat, p_low)
    hi = np.percentile(mat, p_high)
    if hi <= lo:
        hi = float(mat.max())
        lo = float(mat.min())
        if hi <= lo:
            return np.zeros_like(mat, dtype=np.float32)
    out = (mat - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def draw_heatmap_on_image(
    mat,
    img_recover,
    normalize: bool = True,
    p_low: float = 60.0,
    p_high: float = 99.0,
    blur_radius: float = 2.0,
    max_alpha: int = 200,
    alpha_gamma: float = 1.2,
):
    """Render a patch-grid relevancy map as a heatmap overlay.

    Args:
        mat: 2-D numpy array (e.g. 24x24) of per-patch relevancy scores.
        img_recover: PIL.Image background (already de-normalized).
        normalize: whether to percentile-normalize `mat` to [0, 1].
        p_low / p_high: percentile clip bounds; raise p_low to suppress more noise.
        blur_radius: Gaussian blur on the upsampled heatmap (pixels).
        max_alpha: peak opacity of the overlay (0-255).
        alpha_gamma: >1 makes the overlay fall off faster (more of the image shows through).
    """
    mat = np.asarray(mat, dtype=np.float32)
    if normalize:
        mat = _normalize(mat, p_low=p_low, p_high=p_high)
    else:
        mat = np.clip(mat, 0.0, 1.0)

    W, H = img_recover.size

    # Upsample the scalar heat map first (bicubic on a single-channel float image),
    # then apply the colormap - this gives smoother boundaries than upsampling after.
    heat_small = Image.fromarray((mat * 255).astype(np.uint8), mode="L").resize(
        (W, H), Image.BICUBIC
    )
    if blur_radius and blur_radius > 0:
        heat_small = heat_small.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    heat = np.asarray(heat_small, dtype=np.float32) / 255.0

    colored = (cmap(heat)[:, :, :3] * 255).astype(np.uint8)

    # Value-dependent alpha: transparent where heat is low, opaque where heat is high.
    alpha = np.clip(heat**alpha_gamma, 0.0, 1.0)
    alpha = (alpha * max_alpha).astype(np.uint8)

    overlay = Image.fromarray(
        np.dstack([colored, alpha]).astype(np.uint8), mode="RGBA"
    )

    base = img_recover.convert("RGBA")
    composited = Image.alpha_composite(base, overlay)
    return composited.convert("RGB")
