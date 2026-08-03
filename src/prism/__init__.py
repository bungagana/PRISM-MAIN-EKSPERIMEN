"""PRISM: Physics-Informed Reconstruction for Cross-Domain Image Synthesis in
Turbid Media (Sungkono, Muna, et al.).

Three sequential phase-conditional modules, applied as
    I_hat = PCDR(DBLB(PCCA(I_s; phi), I_b; phi); I_cal_t; phi)

- pcca : Phase-Conditional Chromatic Adaptation (Eq. 6)
- dblb : Discriminative Beer-Lambert Blending (Eq. 7-14)
- pcdr : Phase-Conditional Distribution Refinement (Eq. 15-16)
"""
from .pcca import pcca_transform
from .dblb import dblb_blend, compute_kappa
from .pcdr import pcdr_refine
from .pipeline import PrismContext, build_context, synthesize_one, synthesize_phase
from .features import extract_cielab_features

__all__ = [
    "pcca_transform",
    "dblb_blend",
    "compute_kappa",
    "pcdr_refine",
    "PrismContext",
    "build_context",
    "synthesize_one",
    "synthesize_phase",
    "extract_cielab_features",
]
