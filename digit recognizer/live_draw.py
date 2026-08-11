"""
Interactive canvas for testing the trained SVM+HOG digit recognizer by hand.

Run from the project root:
    python -m app.live_draw

Controls
--------
Draw  : hold left mouse button and move
c     : clear canvas
p     : predict
ESC   : quit
"""
import cv2
import numpy as np

from app.preprocess import preprocess_image
from app.predict_svm import predict_digit

CANVAS_SIZE = 600
BRUSH_THICKNESS = 9  # thick loops (6, 8, 9, 0) can fill in and look like solid blobs if this is too high

canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
drawing = False
last_x, last_y = -1, -1


def _draw(event, x, y, flags, param):
    global drawing, last_x, last_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing, last_x, last_y = True, x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, (last_x, last_y), (x, y), (255, 255, 255), BRUSH_THICKNESS, lineType=cv2.LINE_AA)
        last_x, last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def _run_prediction():
    processed = preprocess_image(canvas)
    if processed is None:
        print("Please draw a digit first!")
        return

    cv2.imshow(
        "Processed 28x28",
        cv2.resize(processed, (280, 280), interpolation=cv2.INTER_NEAREST),
    )

    digit, confidence = predict_digit(processed)

    print("=" * 40)
    print(f"Predicted Digit : {digit}")
    if confidence is not None:
        print(f"Confidence      : {confidence * 100:.2f}%")
    else:
        print("Confidence      : n/a (model has no probability output)")
    print("=" * 40)


def main():
    window = "Handwritten Digit Recognizer (SVM + HOG)"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _draw)

    print(__doc__)

    while True:
        cv2.imshow(window, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            canvas[:] = 0
        elif key == ord("p"):
            _run_prediction()
        elif key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
