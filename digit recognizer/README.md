# Handwritten Digit Recognizer (SVM + HOG)

Classifies handwritten digits (0-9) using an SVM trained on HOG
(Histogram of Oriented Gradients) features extracted from MNIST. Includes
a live-drawing demo and a script for testing on your own handwriting.

## Result

| Model                | Features | Test Accuracy |
|-----------------------|-----------|:--------------:|
| **SVM (RBF kernel)** | **HOG**   | **97.5%**      |

## Project structure

```
digit-recognizer/
├── app/                     # importable package used by everything else
│   ├── config.py            # central paths (models, data, outputs)
│   ├── preprocess.py        # canvas/photo -> centered 28x28 digit
│   ├── predict_svm.py       # SVM + HOG inference
│   └── live_draw.py         # interactive drawing demo (OpenCV window)
├── models/
│   └── svm_hog_model.pkl
├── notebooks/
│   └── svm_hog_training.ipynb   # trains & saves the SVM+HOG model
├── scripts/
│   └── test_custom_samples.py   # batch-test on your own handwriting
├── data/samples/             # put your own handwritten digit images here
├── outputs/                  # results (csv) get written here
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Draw and predict live:**

```bash
python -m app.live_draw
```
Controls: draw with the mouse, `c` clear, `p` predict, `Esc` quit.

**Test on your own handwritten samples** (photos or scans):

1. Drop images into `data/samples/`, named so the digit is the first
   character of the filename, e.g. `7_sample1.png`.
2. Run:
   ```bash
   python scripts/test_custom_samples.py
   ```
   This prints a per-image prediction table and an accuracy summary,
   and writes `outputs/custom_sample_results.csv`.

**Retrain the model**: open `notebooks/svm_hog_training.ipynb` in
Jupyter and run all cells — it saves the model straight into `models/`.

## Notes

- `preprocess.py` expects a **white digit on a black background** (like
  the live-draw canvas). `scripts/test_custom_samples.py` auto-detects
  and inverts photos of pen-on-paper digits, which are usually the
  opposite.
- Digits are re-centered by **center of mass**, not just bounding box -
  this matches how the original MNIST dataset itself was constructed,
  and fixes a common source of misclassification for live-drawn digits.
- The saved SVM was trained without `probability=True`, so
  `predict_svm.py` returns `confidence=None`. Retrain with
  `probability=True` if you want a confidence score (training will be
  slower).
- All model/data paths are centralized in `app/config.py`, so scripts
  and notebooks work no matter which directory you launch them from.

## Remaining work

- [ ] Real-world testing on personal handwritten samples (script is ready — add images to `data/samples/`)
- [x] Live-draw demo
- [ ] Final written report
