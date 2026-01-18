from typing import List

from utils.logger import get_logger

logger = get_logger("text_features")


def summarize_text(lines: List[str]) -> List[str]:
    """
    Clean and normalize OCR output for downstream reasoning.
    Removes duplicates, very short noise, and normalizes case.
    """

    if not lines:
        return []

    cleaned = []
    seen = set()

    for line in lines:
        line = line.strip()

        # Skip very short noisy tokens
        if len(line) < 2:
            continue

        norm = line.lower()

        if norm not in seen:
            seen.add(norm)
            cleaned.append(line)

    logger.debug(f"Cleaned OCR lines: {cleaned}")
    return cleaned
