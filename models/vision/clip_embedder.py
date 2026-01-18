import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from typing import List

from config.settings import CLIP_MODEL_NAME, USE_GPU_IF_AVAILABLE
from utils.image_utils import load_image_rgb, resize_preserve_aspect
from utils.logger import get_logger

logger = get_logger("clip_embedder")


class ClipImageEmbedder:
    def __init__(self):
        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")

        self.device = "cuda" if (USE_GPU_IF_AVAILABLE and torch.cuda.is_available()) else "cpu"

        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

        self.model.eval()
        logger.info(f"CLIP running on: {self.device}")

    @torch.no_grad()
    def embed_image(self, image_path: str) -> List[float]:
        """
        Produce a normalized image embedding vector.
        Returns a Python list (JSON-serializable).
        """

        img = load_image_rgb(image_path)
        img = resize_preserve_aspect(img, max_size=1024)

        inputs = self.processor(
            images=img,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.get_image_features(**inputs)

        # Normalize embedding
        embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

        vec = embedding.squeeze(0).cpu().numpy().astype(np.float32)

        return vec.tolist()
