# PRISM — Physics-Informed Reconstruction for Cross-Domain Image Synthesis in Turbid Media

Implementation of PRISM (Sungkono, Muna, et al.) for synthesizing microalgae
growth-phase images in POME (Palm Oil Mill Effluent) medium from clear-water
source images, together with a trained downstream growth-phase classifier.

## Pipeline (paper Algorithm 1)

```
I_hat = PCDR( DBLB( PCCA(I_s; phi), I_b; phi ); I_cal_t; phi )
```

| Module | File | Paper ref |
|---|---|---|
| Phase-Conditional Chromatic Adaptation | `src/prism/pcca.py` | Eq. 6 |
| Discriminative Beer-Lambert Blending | `src/prism/dblb.py` | Eq. 7-14 |
| Phase-Conditional Distribution Refinement | `src/prism/pcdr.py` | Eq. 15-16 |
| Orchestration / calibration context | `src/prism/pipeline.py` | Algorithm 1 |
| 117-dim CIELAB feature extractor | `src/prism/features.py` | Section 3.4 |
| Split builder | `src/prism/data.py` | Section 3.1.2 |

`d_ph(phi)` and `beta_FDA(phi)` follow the calibrated values reported in the
paper (Fig. 3 / Table 2). `kappa_c` and `lambda_c,phi` are estimated from
the calibration images at run time, as specified in Algorithm 1 (Eq. 9-10,
Eq. 15).

## Data availability

Raw photographs (`input/source`, `input/target`, `input/background`) are
**not included in this repository** for data-privacy reasons. To run the
pipeline, organize your own images in the same structure and place them
under `input/`:

```
input/
  source/{lag,log,stationer,death}/*.jpg   # clear-water reference images, per growth phase
  target/{Lag,Log,Stationer,Death}/*.jpg   # POME growth-phase photographs
  background/*.jpg                          # POME background photographs
```

To request the original dataset, contact the corresponding author (see the
paper).

## Setup

```bash
pip install -r requirements.txt
```

## How to run

```bash
cd scripts
python run_all.py
```

runs all three steps below in order. They can also be run individually:

| Step | Command | Produces |
|---|---|---|
| 1. Build calibration/test split | `python 01_build_split.py` | `config/final_split_manifest.json` |
| 2. Synthesize PRISM images | `python 02_synthesize.py` | `output/synthetic/{artifacts,jpg}/` |
| 3. Train + save the classifier | `python 03_train_classifier.py` | `output/classifier/models/best_growth_phase_classifier.joblib` |

No GPU required; a full run (2,000 synthetic images + 6 classifiers) takes
roughly 10-15 minutes on a standard laptop CPU.

## Using the trained classifier

`output/classifier/models/` already contains a trained model, so this step
can be used directly without re-running the pipeline. Only `scripts/predict.py`
+ `best_growth_phase_classifier.joblib` + `model_metadata.json` are needed —
no PRISM synthesis code required at inference time.

```bash
python scripts/predict.py path/to/photo.jpg
python scripts/predict.py path/to/folder_of_photos/
```

```python
from predict import GrowthPhaseClassifier

clf = GrowthPhaseClassifier()
label, confidence, probs = clf.predict_path("sample.jpg")
# label: one of "Lag" / "Log" / "Stationary" / "Death"
```

Dependencies for `predict.py` alone: `numpy`, `scipy`, `opencv-python`,
`joblib`, `scikit-learn`.

## Try it in a browser (demo web app)

```bash
pip install streamlit
streamlit run webapp/app.py
```

Opens at `http://localhost:8501`. Upload a photo, or pick one of the
bundled sample images (`webapp/samples/`, PRISM-synthesized), to see the
predicted growth phase and per-class confidence.

## Reproducibility statement

Following the reproducibility principles of Huettmann and Arhonditsis
(2023, *Ecological Informatics* 76, 102132), this repository provides:

- Complete source code for all three PRISM modules and the downstream
  classifier.
- Fixed random seeds (`prism.config.SEED = 42`, `SPLIT_SEED = 165`).
- An explicit dependency list (`requirements.txt`).
- The exact calibration/test file list used (`config/final_split_manifest.json`).
- The trained classifier and the synthesized images it was trained on.

## Project layout

```
PRISM-MAIN-GITHUB/
├── input/                    # [not in repo — see "Data availability"]
├── config/final_split_manifest.json
├── src/prism/                 # the PRISM package (importable)
│   ├── pcca.py, dblb.py, pcdr.py       # the 3 PRISM modules (Eq. 6, 7-14, 15-16)
│   ├── pipeline.py, data.py            # orchestration + calibration context
│   ├── features.py                      # downstream feature extraction
│   └── config.py                        # parameters/constants
├── scripts/
│   ├── 01_build_split.py, 02_synthesize.py, 03_train_classifier.py
│   ├── run_all.py
│   └── predict.py               # standalone inference helper
├── webapp/                    # Streamlit demo app + sample images
├── output/
│   ├── synthetic/jpg/           # PRISM-synthesized images
│   └── classifier/models/        # best_growth_phase_classifier.joblib + metadata
├── requirements.txt
├── LICENSE
└── .gitignore
```

## License

MIT License — see [LICENSE](LICENSE).
