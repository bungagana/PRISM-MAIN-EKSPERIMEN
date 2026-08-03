"""Run PRISM synthesis (PCCA -> DBLB -> PCDR) for all four growth phases,
saving both a .npy array per phase and individual .jpg previews.
"""
import _bootstrap  # noqa: F401
import json
import time

import numpy as np

from prism import config
from prism.data import save_rgb
from prism.pipeline import build_context, synthesize_phase

OUT_DIR = config.OUTPUT_DIR / "synthetic"
ART_DIR = OUT_DIR / "artifacts"
JPG_DIR = OUT_DIR / "jpg"


def main():
    ART_DIR.mkdir(parents=True, exist_ok=True)
    JPG_DIR.mkdir(parents=True, exist_ok=True)

    print("Building calibration context (PCCA moments, kappa, Yeo-Johnson lambda)...")
    ctx = build_context()
    print("kappa (R,G,B):", ctx.kappa.tolist())
    print("d_ph:", ctx.d_phase)
    print("beta_FDA:", ctx.beta_fda)
    print("lambda (per phase, [L*,a*,b*]):", json.dumps(ctx.lambdas, indent=2))

    synthetic_files = {}
    for phase in config.PHASES:
        phase_dir = JPG_DIR / config.TARGET_PHASE_DIR[phase]
        phase_dir.mkdir(parents=True, exist_ok=True)
        arr_path = ART_DIR / f"prism_{phase.lower()}.npy"

        t0 = time.time()
        images = synthesize_phase(ctx, phase, config.N_SYNTH_PER_PHASE)
        out = np.lib.format.open_memmap(
            arr_path, mode="w+", dtype=np.float32,
            shape=(len(images), config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
        )
        for i, img in enumerate(images):
            out[i] = img
            save_rgb(phase_dir / f"{config.TARGET_PHASE_DIR[phase].lower()}_{i + 1:04d}.jpg", img)
        out.flush()
        del out

        synthetic_files[phase] = str(arr_path)
        print(f"[{phase}] {len(images)} images in {time.time() - t0:.0f}s -> {arr_path}")

    manifest = {
        "image_size": config.IMAGE_SIZE,
        "n_synthetic_per_phase": config.N_SYNTH_PER_PHASE,
        "seed": config.SEED,
        "classes": config.PHASES,
        "synthetic_files": synthetic_files,
        "d_phase": dict(config.D_PHASE),
        "beta_fda": dict(config.BETA_FDA),
        "kappa_mean": ctx.kappa.tolist(),
        "yeojohnson_lambda": ctx.lambdas,
    }
    manifest_path = ART_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print("Saved manifest:", manifest_path)


if __name__ == "__main__":
    main()
