"""Discriminative Beer-Lambert Blending (DBLB) - paper Section 3.3.2, Eq. 7-14.

Models wavelength-dependent light attenuation in POME using the Beer-Lambert
law as a per-sRGB-channel parametric blending kernel between the
PCCA-aligned source image and the POME background image.
"""
import numpy as np
from skimage import color


def compute_kappa(background_images):
    """Eq. 9-10: per-channel attenuation coefficient kappa_c from background
    images, kappa_c = -ln(mean normalized intensity), averaged across all
    available background photographs (Table 3's "Global Mean").
    """
    per_image = np.array([
        -np.log(np.clip(bg.reshape(-1, 3).mean(0), 1e-6, 1.0))
        for bg in background_images
    ])
    return per_image.mean(0), per_image


def dblb_blend(source_lab, background_rgb, kappa, d_ph, beta_spatial=0.15):
    """Eq. 11-14: blend the PCCA-aligned source (converted back to sRGB) with
    the POME background, using Beer-Lambert transmittance modulated by a
    local-contrast map so heterogeneous background texture is preserved.
    """
    source_rgb = np.clip(color.lab2rgb(source_lab), 0, 1)

    omega_c = np.exp(-kappa * d_ph)  # Eq. 11
    bg_mean = background_rgb.mean((0, 1), keepdims=True)
    bg_std = background_rgb.std((0, 1), keepdims=True) + 1e-6
    local_contrast = np.abs(background_rgb - bg_mean) / bg_std  # Eq. 13

    blend_weight = np.clip(omega_c * (1 + beta_spatial * local_contrast), 0, 1)  # Eq. 12
    return (1 - blend_weight) * background_rgb + blend_weight * source_rgb  # Eq. 14
