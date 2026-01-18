import pytesseract
import cv2
from typing import List

from config.settings import TESSERACT_LANG
from utils.image_utils import load_image_bgr
from utils.logger import get_logger

logger = get_logger("ocr_extractor")


class OCRExtractor:
    def __init__(self):
        logger.info("Initializing Tesseract OCR")

    def extract_text(self, image_path: str) -> List[str]:
        """
        Extract readable text lines from image.
        Returns list of cleaned text snippets.
        """

        img_bgr = load_image_bgr(image_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Improve OCR quality
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        raw_text = pytesseract.image_to_string(gray, lang=TESSERACT_LANG)

        # Clean and split lines
        lines = [
            line.strip()
            for line in raw_text.split("\n")
            if line.strip()
        ]

        logger.debug(f"OCR extracted {len(lines)} lines")
        return lines
