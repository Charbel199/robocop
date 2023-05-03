

from paddleocr import PaddleOCR
import cv2
import numpy as np
import itertools
from operator import itemgetter
import re
paddle = PaddleOCR(lang="en", cls_model_dir='./assets/cls', det_model_dir='./assets/det/en', rec_model_dir='./assets/rec/en')


def ocr_plate(plate_region):
    # Image pre-processing for more accurate OCR
    rescaled = cv2.resize(
        plate_region, (700,225))
    grayscale = cv2.cvtColor(rescaled, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(grayscale, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # OCR the preprocessed image
    results = paddle.ocr(eroded, det=False, cls=False)
    flattened = list(itertools.chain.from_iterable(results))
    plate_text, ocr_confidence = max(flattened, key=itemgetter(1), default=("", 0))

    # Filter out anything but uppercase letters, digits, hypens and whitespace.
    plate_text = re.sub(r"[^-A-Z0-9 ]", r"", plate_text).strip()

    if ocr_confidence == "nan":
        ocr_confidence = 0

    return plate_text, ocr_confidence


if __name__ == "__main__":
    img = cv2.imread('/home/charbel199/Downloads/plate2.jpeg')
    a, _ = ocr_plate(img)
    print(a)