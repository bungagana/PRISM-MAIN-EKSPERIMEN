"""Phase-Conditional Distribution Refinement (PCDR) - paper Section 3.3.3,
Eq. 15-16.

Two steps, both conditioned on growth phase:
1. Yeo-Johnson transform to correct higher-order distribution moments
   (skewness/kurtosis), moment-matched to the target calibration set in the
   transformed space, then mapped back to CIELAB with the analytic inverse
   transform (scipy.stats.yeojohnson only provides the forward direction).
2. Multiscale Fourier Domain Adaptation (fine/medium/coarse) fused with
   Gaussian weights {0.25, 0.50, 0.25} around the calibrated beta_phi.
"""
import numpy as np
from scipy import stats
from skimage import color


def fit_yeojohnson_lambda(calibration_pixels):
    """Algorithm 1 line 5: fit lambda_c,phi via MLE, per CIELAB channel, on
    the target calibration pixel pool (NOT a fixed constant - this is a
    dataset-dependent shape-normalisation parameter, re-estimated for
    whatever target_cal images are supplied).
    """
    return [float(stats.yeojohnson_normmax(calibration_pixels[:, ch])) for ch in range(3)]


def _inverse_yeojohnson(y, lam, eps=1e-6):
    """Analytic inverse of scipy.stats.yeojohnson's forward transform.

    scipy only implements the forward direction. Eq. 15 requires values to
    be mapped back into CIELAB units before the FDA stage (which operates on
    real CIELAB images, Algorithm 1 line 20), so the inverse must be applied
    after moment-matching in the transformed space.
    """
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    pos = y >= 0
    neg = ~pos
    if abs(lam) > eps:
        base = np.clip(y[pos] * lam + 1.0, eps, None)
        out[pos] = np.power(base, 1.0 / lam) - 1.0
    else:
        out[pos] = np.expm1(y[pos])
    if abs(lam - 2.0) > eps:
        base = np.clip(-(2.0 - lam) * y[neg] + 1.0, eps, None)
        out[neg] = 1.0 - np.power(base, 1.0 / (2.0 - lam))
    else:
        out[neg] = 1.0 - np.exp(-y[neg])
    return out


def yeojohnson_moment_match(image_lab, calibration_pixels, lambdas, max_gain=1.5):
    """Eq. 15: forward YJ -> moment-match to target_cal (in transformed
    space) -> inverse YJ back to CIELAB, per channel.

    The rescale factor r.std()/q.std() is bounded by max_gain for the same
    reason as PCCA's transform matrix (see pcca.compute_transform_matrix):
    a single source image's per-channel spread in the Yeo-Johnson-transformed
    space can be small relative to the calibration pool's spread, and an
    unconstrained rescale can over-amplify fine-grained noise. Same
    shrinkage/regularization rationale and margin as PCCA's max_gain.
    """
    out = np.empty_like(image_lab)
    for ch, lam in enumerate(lambdas):
        v = image_lab[..., ch].ravel().astype(np.float64)
        ref = calibration_pixels[:, ch].astype(np.float64)
        q = stats.yeojohnson(v, lam)
        r = stats.yeojohnson(ref, lam)
        gain = min(r.std() / (q.std() + 1e-6), max_gain)
        matched = (q - q.mean()) * gain + r.mean()
        back = _inverse_yeojohnson(matched, lam)
        out[..., ch] = np.clip(
            back, np.quantile(ref, 0.005), np.quantile(ref, 0.995)
        ).reshape(image_lab.shape[:2])
    return out


def fourier_domain_adapt(source_lab, reference_rgb, beta):
    """Single-scale FDA: replace the low-frequency amplitude band (of size
    proportional to beta) of the source with the reference's amplitude,
    keeping the source's phase (so structural/spatial layout is preserved).
    """
    src_fft = np.fft.fftshift(np.fft.fft2(source_lab, axes=(0, 1)), axes=(0, 1))
    ref_fft = np.fft.fftshift(np.fft.fft2(color.rgb2lab(reference_rgb), axes=(0, 1)), axes=(0, 1))
    h, w = source_lab.shape[:2]
    band = max(1, int(min(h, w) * beta / 2))
    cy, cx = h // 2, w // 2

    amp = np.abs(src_fft)
    amp[cy - band:cy + band, cx - band:cx + band] = np.abs(ref_fft)[cy - band:cy + band, cx - band:cx + band]
    phase = np.angle(src_fft)
    return np.real(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * phase), axes=(0, 1)), axes=(0, 1)))


def multiscale_fda(source_lab, reference_rgb, beta):
    """Eq. 16: fuse fine/medium/coarse scale FDA with weights {0.25, 0.5, 0.25}."""
    return (
        0.25 * fourier_domain_adapt(source_lab, reference_rgb, max(0.01, beta - 0.02))
        + 0.50 * fourier_domain_adapt(source_lab, reference_rgb, beta)
        + 0.25 * fourier_domain_adapt(source_lab, reference_rgb, min(0.20, beta + 0.02))
    )


def pcdr_refine(dblb_output_rgb, calibration_pixels, lambdas, reference_rgb, beta):
    """Full PCDR stage: Yeo-Johnson moment-matching then multiscale FDA,
    returning a clipped sRGB image (Algorithm 1 lines 16-21).
    """
    lab = color.rgb2lab(dblb_output_rgb)
    yj = yeojohnson_moment_match(lab, calibration_pixels, lambdas)
    fused_lab = multiscale_fda(yj, reference_rgb, beta)
    return np.clip(color.lab2rgb(fused_lab), 0, 1).astype(np.float32)
