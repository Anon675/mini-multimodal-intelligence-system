import argparse
import json
from pathlib import Path
import time

from dotenv import load_dotenv

from pipeline.inference_pipeline import InferencePipeline
from utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Mini Multimodal Intelligence System — Image Reasoning Pipeline"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Absolute path to folder with input images",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Absolute path to folder where JSON outputs will be saved",
    )

    args = parser.parse_args()

    load_dotenv()
    logger = get_logger("main")

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # ---------- SANITY CHECKS ----------
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Invalid input folder: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- ROBUST IMAGE DISCOVERY ----------
    print("\nDEBUG: Listing contents of inputs folder:")
    for f in input_dir.iterdir():
        print("  -", f.name)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ]

    if len(image_paths) == 0:
        raise ValueError(f"No valid images found in {input_dir}")

    print(f"\nDEBUG: Found {len(image_paths)} valid images:")
    for p in image_paths:
        print("  -", p)

    # ---------- INITIALIZE PIPELINE WITH TIMING ----------
    logger.info(">>> About to create InferencePipeline()")
    t0 = time.time()

    pipeline = InferencePipeline()

    logger.info(f">>> Pipeline created in {time.time() - t0:.2f} seconds")

    # ---------- PROCESS IMAGES (WITH FAIL-SAFE WRITING) ----------
    logger.info(f">>> Processing {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths, start=1):
        logger.info(f"[{i}/{len(image_paths)}] Running on: {img_path.name}")

        try:
            result = pipeline.run(str(img_path))

        except Exception as e:
            logger.error(
                f"PIPELINE FAILED on {img_path.name} — writing fallback JSON",
                exc_info=True,
            )
            # GUARANTEED OUTPUT EVEN IF LLM FAILS
            result = {
                "image_quality_score": None,
                "issues_detected": ["pipeline_error"],
                "detected_objects": [],
                "text_detected": [],
                "llm_reasoning_summary": str(e),
                "final_verdict": "Unknown",
                "confidence": 0.0,
            }

        out_path = output_dir / f"{img_path.stem}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved: {out_path}")

    logger.info(">>> All images processed.")


if __name__ == "__main__":
    main()
