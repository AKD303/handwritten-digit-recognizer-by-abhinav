"""
Turn a raw canvas/photo into a centered 28x28 grayscale digit image,
matching the format the SVM+HOG model was trained on (white digit,
black background, mass-centered like real MNIST images).
"""
import cv2
import numpy as np


def preprocess_image(canvas: np.ndarray) -> np.ndarray | None:
    """
    Parameters
    ----------
    canvas : np.ndarray
        BGR (or already-grayscale) image containing a white digit on a
        black background.

    Returns
    -------
    np.ndarray of shape (28, 28), dtype uint8, or None if no digit was
    found in the image.
    """
    if canvas.ndim == 3:
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    else:
        gray = canvas

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    if w == 0 or h == 0:
        return None

    # Small padding so thick strokes (loops, tails) aren't clipped flush
    # against the crop edge.
    pad = max(2, int(0.05 * max(w, h)))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1 = min(x + w + pad, thresh.shape[1])
    y1 = min(y + h + pad, thresh.shape[0])
    digit = thresh[y0:y1, x0:x1]

    h, w = digit.shape

    # Preserve aspect ratio, longest side -> 20px (leaves a border, like
    # the original MNIST preprocessing pipeline).
    if h > w:
        new_h = 20
        new_w = max(1, int(w * 20 / h))
    else:
        new_w = 20
        new_h = max(1, int(h * 20 / w))

    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Soften the resized strokes slightly - MNIST digits are naturally
    # anti-aliased, whereas a mouse-drawn canvas has flat, hard edges.
    digit = cv2.GaussianBlur(digit, (3, 3), 0)

    output = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

    # Re-center by center of mass, not just bounding box - this is how
    # the original MNIST dataset was constructed, and skipping it is a
    # common reason live-drawn digits get misclassified even when the
    # model tests well on real MNIST data.
    output = _center_by_mass(output)

    return output


def _center_by_mass(image: np.ndarray) -> np.ndarray:
    moments = cv2.moments(image)
    if moments["m00"] == 0:
        return image

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    shift_x = int(round(image.shape[1] / 2.0 - cx))
    shift_y = int(round(image.shape[0] / 2.0 - cy))

    shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, shift_matrix, (image.shape[1], image.shape[0]))
