"""Orchestrates the full PRISM synthesis: PCCA -> DBLB -> PCDR (Algorithm 1).

    I_hat = PCDR(DBLB(PCCA(I_s; phi), I_b; phi); I_cal_t; phi)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config
from .data import load_split_manifest, list_images, read_rgb
from .dblb import compute_kappa, dblb_blend
from .pcca import pcca_transform, phase_moments, sample_lab_pixels
from .pcdr import fit_yeojohnson_lambda, pcdr_refine


@dataclass
class PrismContext:
    rng: np.random.Generator
    source_images: Dict[str, List[np.ndarray]]
    target_cal_images: Dict[str, List[np.ndarray]]
    background_images: List[np.ndarray]
    kappa: np.ndarray
    kappa_per_background: np.ndarray
    source_moments: Dict[str, tuple]
    target_moments: Dict[str, tuple]
    calibration_pixels: Dict[str, np.ndarray]
    lambdas: Dict[str, list]
    d_phase: Dict[str, float] = field(default_factory=lambda: dict(config.D_PHASE))
    beta_fda: Dict[str, float] = field(default_factory=lambda: dict(config.BETA_FDA))


def build_context(input_dir=config.INPUT_DIR, split_path=None, seed=config.SEED,
                   target_cal_cap=config.TARGET_CAL_MOMENT_CAP):
    """Load source/target/background images and estimate all calibration
    statistics (PCCA moments, kappa, Yeo-Johnson lambda) from target_cal.
    Nothing here touches target_test or any published headline metric.
    """
    input_dir = Path(input_dir)
    rng = np.random.default_rng(seed)
    split = load_split_manifest(split_path)
    project_root = input_dir.parent

    source_images, target_cal_images, calibration_pixels = {}, {}, {}
    for phase in config.PHASES:
        source_paths = list_images(input_dir / "source" / config.SOURCE_PHASE_DIR[phase])
        source_images[phase] = [read_rgb(p) for p in source_paths]

        cal_paths = sorted(project_root / p for p in split["target_cal"][phase])[:target_cal_cap]
        target_cal_images[phase] = [read_rgb(p) for p in cal_paths]
        calibration_pixels[phase] = sample_lab_pixels(target_cal_images[phase], rng, 40000)

    background_paths = list_images(input_dir / "background")
    background_images = [read_rgb(p) for p in background_paths]
    kappa, kappa_per_background = compute_kappa(background_images)

    source_moments = {p: phase_moments(source_images[p], rng) for p in config.PHASES}
    target_moments = {p: phase_moments(target_cal_images[p], rng) for p in config.PHASES}
    lambdas = {p: fit_yeojohnson_lambda(calibration_pixels[p]) for p in config.PHASES}

    return PrismContext(
        rng=rng, source_images=source_images, target_cal_images=target_cal_images,
        background_images=background_images, kappa=kappa, kappa_per_background=kappa_per_background,
        source_moments=source_moments, target_moments=target_moments,
        calibration_pixels=calibration_pixels, lambdas=lambdas,
    )


def synthesize_one(ctx: PrismContext, phase, source_rgb, background_rgb, image_size=config.IMAGE_SIZE):
    """Run Algorithm 1 stages 1-3 for a single (source, background) pair."""
    mu_s, cov_s = ctx.source_moments[phase]
    mu_t, cov_t = ctx.target_moments[phase]
    lab_aligned = pcca_transform(source_rgb, mu_s, cov_s, mu_t, cov_t, image_size)

    dblb_out = dblb_blend(lab_aligned, background_rgb, ctx.kappa, ctx.d_phase[phase], config.BETA_SPATIAL)

    reference_rgb = ctx.target_cal_images[phase][ctx.rng.integers(len(ctx.target_cal_images[phase]))]
    return pcdr_refine(
        dblb_out, ctx.calibration_pixels[phase], ctx.lambdas[phase], reference_rgb, ctx.beta_fda[phase]
    )


def synthesize_phase(ctx: PrismContext, phase, n_images=config.N_SYNTH_PER_PHASE):
    """Generate n_images synthetic images for one phase, cycling through the
    available source images and background photographs.
    """
    sources = ctx.source_images[phase]
    backgrounds = ctx.background_images
    images = []
    for i in range(n_images):
        src = sources[i % len(sources)]
        bg = backgrounds[(i // len(sources)) % len(backgrounds)]
        images.append(synthesize_one(ctx, phase, src, bg))
    return images
