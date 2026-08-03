"""Run the full pipeline end to end: split -> synthesize -> train classifier."""
import _bootstrap  # noqa: F401
import runpy
from pathlib import Path

STEPS = [
    "01_build_split.py",
    "02_synthesize.py",
    "03_train_classifier.py",
]


def main():
    here = Path(__file__).resolve().parent
    for step in STEPS:
        print("\n" + "=" * 70)
        print("RUNNING:", step)
        print("=" * 70)
        runpy.run_path(str(here / step), run_name="__main__")


if __name__ == "__main__":
    main()
