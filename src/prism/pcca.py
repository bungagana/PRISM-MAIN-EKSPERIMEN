"""Phase-Conditional Chromatic Adaptation (PCCA) - paper Section 3.3.1, Eq. 6.

    x'_1 = Sigma_t(phi)^(1/2) . Sigma_s(phi)^(-1/2) . (x - mu_s(phi)) + mu_t(phi)

Second-order (mean + covariance) statistical transform in CIELAB space,
computed per growth phase to preserve the correlation between L*, a*, b*
(especially L*-a*, which reflects chlorophyll-related color variation).
"""
import numpy as np
from scipy.linalg import sqrtm
from skimage import color


def phase_moments(images, rng, n_pixels=35000):
    """Sample CIELAB pixels across a set of images and return (mean, cov).

    A small ridge (1e-5) is added to the covariance diagonal for numerical
    stability when inverting sqrtm(cov) - source phases have as few as 3
    images, so the raw sample covariance can be near-singular.
    """
    pixels = sample_lab_pixels(images, rng, n_pixels)
    mean = pixels.mean(0)
    cov = np.cov(pixels.T) + np.eye(3) * 1e-5
    return mean, cov


def sample_lab_pixels(images, rng, n_pixels):
    chunks = []
    per_image = max(1, int(np.ceil(n_pixels / max(1, len(images)))))
    for img in images:
        lab = color.rgb2lab(img).reshape(-1, 3).astype(np.float32, copy=False)
        take = min(per_image, len(lab))
        idx = rng.choice(len(lab), take, replace=False)
        chunks.append(lab[idx])
    pixels = np.vstack(chunks)
    if len(pixels) > n_pixels:
        pixels = pixels[rng.choice(len(pixels), n_pixels, replace=False)]
    return pixels


def compute_transform_matrix(cov_s, cov_t, max_gain=1.5):
    """Eq. 6's linear map A = Sigma_t^(1/2) . Sigma_s^(-1/2), with its
    singular values bounded by max_gain.

    Rationale: source phases are represented by a small number of images
    (3-11), so the sample covariance cov_s can be poorly conditioned along
    some CIELAB direction relative to the calibration pool's covariance
    cov_t. Under an unconstrained whitening-recoloring map, this can
    over-amplify fine-grained sensor/compression noise present in the
    source image. Bounding the map's spectral norm (its largest singular
    value) is a standard shrinkage/regularization technique for
    covariance-based color transfer (cf. Pitie et al. on numerical
    conditioning in linear Monge/covariance color transfer): it preserves
    Eq. 6's mean/covariance realignment for well-conditioned directions
    while constraining the transform where the source covariance is small.
    max_gain=1.5 was selected empirically to leave typical well-conditioned
    cases unaffected while keeping the output visually clean.
    """
    a = np.real(sqrtm(cov_t)) @ np.linalg.inv(np.real(sqrtm(cov_s)))
    u, s, vt = np.linalg.svd(a)
    s_clipped = np.clip(s, None, max_gain)
    return u @ np.diag(s_clipped) @ vt


def pcca_transform(image_rgb, mu_s, cov_s, mu_t, cov_t, image_size, max_gain=1.5):
    """Apply Eq. 6 to a single sRGB image, returning a CIELAB array."""
    a = compute_transform_matrix(cov_s, cov_t, max_gain=max_gain)
    lab = color.rgb2lab(image_rgb).reshape(-1, 3)
    transformed = (lab - mu_s) @ a.T + mu_t
    return transformed.reshape(image_size, image_size, 3)
