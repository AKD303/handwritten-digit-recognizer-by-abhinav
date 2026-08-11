"""
SVM + HOG inference on a preprocessed 28x28 digit image.

The HOG parameters here MUST match notebooks/svm_hog_training.ipynb exactly
(orientations, pixels_per_cell, cells_per_block, block_norm) or the feature
vector shape/meaning won't line up with what the SVM was trained on.
"""
import numpy as np
import joblib
from skimage.feature import hog

from app.config import SVM_MODEL_PATH

_model = None

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(SVM_MODEL_PATH)
    return _model


def _extract_hog(image: np.ndarray) -> np.ndarray:
    return hog(image, **HOG_PARAMS)


def predict_digit(processed_image: np.ndarray) -> tuple[int, float | None]:
    """
    Parameters
    ----------
    processed_image : np.ndarray, shape (28, 28)

    Returns
    -------
    (digit, confidence) : (int, float | None)
        confidence is None because the saved SVC was trained without
        `probability=True`, so no calibrated probability is available.
        Retrain with `SVC(..., probability=True)` if you want one -
        it will make training noticeably slower.
    """
    model = _get_model()

    features = _extract_hog(processed_image).reshape(1, -1)
    digit = int(model.predict(features)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)))

    return digit, confidence
