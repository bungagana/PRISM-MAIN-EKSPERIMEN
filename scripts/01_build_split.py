"""Build config/final_split_manifest.json (dominant-batch-per-phase split)."""
import _bootstrap  # noqa: F401
import json

from prism import config
from prism.data import build_split_manifest


def main():
    manifest = build_split_manifest()
    for phase in config.PHASES:
        n_cal = len(manifest["target_cal"][phase])
        n_test = len(manifest["target_test"][phase])
        print(f"{phase:12s} dominant_batch={manifest['dominant_batch'][phase]:>3s} "
              f"-> {n_cal} cal / {n_test} test")
    out_path = config.CONFIG_DIR / "final_split_manifest.json"
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
