import cv2
import pytesseract
import numpy as np
from typing import Optional, List

from config.settings import TESSERACT_LANG, TESSERACT_TIMEOUT_SEC
from utils.logger import get_logger


class OCRExtractor:
    """
    OCR extractor specialized for:
    - large signage text (SALE, DISCOUNT, OFF)
    - high-contrast boards (red/white)
    - retail environments
    """

    def __init__(self):
        self.logger = get_logger("ocr_extractor")

        try:
            pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError("Tesseract OCR not available in PATH")

    # ---------------------------------------------------------
    # STEP 1: LOCATE RED SIGN BOARDS (SALE BOARDS)
    # ---------------------------------------------------------
    def _extract_red_regions(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = mask1 | mask2

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            area = w * h
            img_area = image_bgr.shape[0] * image_bgr.shape[1]

            # Only keep large boards
            if area > 0.05 * img_area:
                region = image_bgr[y:y + h, x:x + w]
                regions.append(region)

        return regions

    # ---------------------------------------------------------
    # STEP 2: OCR ON CROPPED SIGN
    # ---------------------------------------------------------
    def _ocr_on_region(self, region: np.ndarray) -> str:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        config = "--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        text = pytesseract.image_to_string(
            thresh,
            lang=TESSERACT_LANG,
            config=config,
            timeout=TESSERACT_TIMEOUT_SEC,
        )

        return text.strip()

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------
    def extract_text(self, image_path: str) -> Optional[str]:
        image = cv2.imread(image_path)

        if image is None:
            return ""

        detected_text = []

        red_regions = self._extract_red_regions(image)

        for region in red_regions:
            text = self._ocr_on_region(region)
            if len(text) >= 3:
                detected_text.append(text)

        if detected_text:
            final_text = " ".join(detected_text)
            self.logger.info(f"OCR detected text: {final_text}")
            return final_text

        return ""
