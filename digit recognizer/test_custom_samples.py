"""
Evaluate the trained SVM+HOG model on your own handwritten digit
photos/scans — this is the "real-world testing" step.

USAGE
-----
1. Put handwritten digit images in data/samples/
2. Name each file starting with its true digit, e.g. "7_photo1.png",
   "3.jpg", "9-test.png". Only the FIRST character of the filename needs
   to be the digit; the rest is ignored, so any naming scheme works as
   long as it starts with the digit.
3. From the project root, run:
       python scripts/test_custom_samples.py

Images can be photos of pen-on-paper digits (dark digit on light
background) or drawn white-on-black — this script auto-detects and
inverts if needed.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SAMPLES_DIR, OUTPUTS_DIR
from app.preprocess import preprocess_image
from app.predict_svm import predict_digit

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def load_and_normalize(path: Path) -> np.ndarray:
    """Load an image and make sure it's white-digit-on-black, like the
    live-draw canvas the model was tuned against."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127:  # mostly light background -> likely pen-on-paper
        img = cv2.bitwise_not(img)

    return img


def true_label_from_filename(path: Path):
    first_char = path.stem[0]
    return int(first_char) if first_char.isdigit() else None


def main():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in {SAMPLES_DIR}")
        print("Add some handwritten digit images (see the docstring at the "
              "top of this file for naming rules) and re-run this script.")
        return

    rows = []
    correct = labeled_count = 0

    header = f"{'File':<28}{'True':<6}{'Predicted':<10}"
    print(header)
    print("-" * len(header))

    for path in image_paths:
        canvas = load_and_normalize(path)
        processed = preprocess_image(canvas)

        if processed is None:
            print(f"{path.name:<28} could not find a digit in this image, skipping")
            continue

        pred_digit, _ = predict_digit(processed)
        true_label = true_label_from_filename(path)

        if true_label is not None:
            labeled_count += 1
            correct += int(pred_digit == true_label)

        true_str = str(true_label) if true_label is not None else "-"
        print(f"{path.name:<28}{true_str:<6}{pred_digit:<10}")

        rows.append({
            "file": path.name,
            "true_label": true_label,
            "predicted": pred_digit,
        })

    if labeled_count:
        print("-" * len(header))
        print(f"Accuracy on {labeled_count} labeled samples: "
              f"{correct / labeled_count * 100:.2f}%")

    if rows:
        out_csv = OUTPUTS_DIR / "custom_sample_results.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
