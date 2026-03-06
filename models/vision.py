import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from PIL import ImageOps
from transformers import CLIPProcessor, CLIPModel
import importlib
from collections import OrderedDict

pillow_heif = None
try:
    pillow_heif = importlib.import_module("pillow_heif")
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except Exception:
    HEIF_AVAILABLE = False


class VisionEncoder:
    _did_log_device = False

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        self.device = self._resolve_device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self._text_cache_max_size = 256
        self._text_emb_lru: OrderedDict[tuple[str, ...], np.ndarray] = OrderedDict()
        if not VisionEncoder._did_log_device:
            print(f"[VisionEncoder] Using device: {self.device}")
            VisionEncoder._did_log_device = True

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        # Priority: explicit arg > cuda > mps > cpu
        if device is not None:
            d = str(device).strip().lower()
            if d == "cuda" and not torch.cuda.is_available():
                raise ValueError("Requested device='cuda' but CUDA is not available.")
            if d == "mps" and not torch.backends.mps.is_available():
                raise ValueError("Requested device='mps' but MPS is not available.")
            return d
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _move_inputs_to_device(self, inputs: dict) -> dict:
        out = {}
        for k, v in inputs.items():
            out[k] = v.to(self.device) if hasattr(v, "to") else v
        return out

    def _open_image(self, image_path: str) -> Image.Image:
        try:
            img = Image.open(image_path)
        except UnidentifiedImageError as exc:
            if HEIF_AVAILABLE and pillow_heif is not None:
                # Fallback for HEIC content with wrong extension (e.g. temp .jpg filename).
                try:
                    heif = pillow_heif.read_heif(image_path)
                    img = Image.frombytes(
                        heif.mode, heif.size, heif.data, "raw", heif.mode, heif.stride
                    )
                except Exception:
                    pass
            if "img" not in locals():
                if str(image_path).lower().endswith((".heic", ".heif")) and not HEIF_AVAILABLE:
                    raise RuntimeError(
                        "HEIC/HEIF image detected, but HEIF support is missing. "
                        "Install dependency: pip install pillow-heif"
                    ) from exc
                raise
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")

    @staticmethod
    def _zoom_in_crop(img: Image.Image, scale: float = 0.9) -> Image.Image:
        w, h = img.size
        nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
        left = (w - nw) // 2
        top = (h - nh) // 2
        return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.Resampling.BICUBIC)

    @staticmethod
    def _zoom_out_pad(img: Image.Image, scale: float = 1.1) -> Image.Image:
        w, h = img.size
        nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
        canvas = Image.new("RGB", (nw, nh), (0, 0, 0))
        left = (nw - w) // 2
        top = (nh - h) // 2
        canvas.paste(img, (left, top))
        return canvas.resize((w, h), Image.Resampling.BICUBIC)

    @torch.no_grad()
    def _encode_pil_image(self, img: Image.Image) -> np.ndarray:
        # Single canonical preprocessing path for all image embeddings.
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs)
        pixel_values = inputs["pixel_values"]

        vision_out = self.model.vision_model(pixel_values=pixel_values)
        pooled = getattr(vision_out, "pooler_output", None)
        if pooled is None:
            pooled = vision_out.last_hidden_state[:, 0, :]

        proj = getattr(self.model, "visual_projection", None)
        feats = proj(pooled) if proj is not None else pooled
        vec = feats.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)

    @torch.no_grad()
    def encode_image(self, image_path: str, multi_crop: bool = False) -> np.ndarray:
        """
        Return a single L2-normalized image embedding (shape: [D], typically D=512).

        Newer transformers versions may return patch tokens from `get_image_features()`.
        To keep this stable, we explicitly:
        - run the vision tower
        - take pooled output (or CLS token fallback)
        - apply the model's `visual_projection` (if present)
        """
        img = self._open_image(image_path)
        if not multi_crop:
            return self._encode_pil_image(img)

        views = [
            img,
            self._zoom_in_crop(img, scale=0.9),
            self._zoom_out_pad(img, scale=1.1),
        ]
        embs = [self._encode_pil_image(v) for v in views]
        avg = np.mean(np.vstack(embs), axis=0).astype(np.float32)
        return avg / (np.linalg.norm(avg) + 1e-12)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        Return L2-normalized text embeddings (shape: [N, D], typically D=512).
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = self._move_inputs_to_device(inputs)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        text_out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(text_out, "pooler_output", None)
        if pooled is None:
            pooled = text_out.last_hidden_state[:, 0, :]

        proj = getattr(self.model, "text_projection", None)
        feats = proj(pooled) if proj is not None else pooled
        arr = feats.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms

    @torch.no_grad()
    def encode_texts_cached(self, prompts: list[str]) -> np.ndarray:
        """
        Cached text embeddings for repeated prompt lists.
        LRU key: tuple(prompts)
        """
        key = tuple(prompts)
        if key in self._text_emb_lru:
            arr = self._text_emb_lru.pop(key)
            self._text_emb_lru[key] = arr
            return arr
        arr = self.encode_texts(list(key))
        self._text_emb_lru[key] = arr
        if len(self._text_emb_lru) > self._text_cache_max_size:
            self._text_emb_lru.popitem(last=False)
        return arr

    @torch.no_grad()
    def score_image_prompts_from_emb(self, image_emb: np.ndarray, prompts: list[str]) -> np.ndarray:
        """
        Cosine similarities between a precomputed image embedding and prompt list.
        Uses cached text embeddings for repeated prompt sets.
        """
        img = np.asarray(image_emb, dtype=np.float32).reshape(-1)
        img = img / (np.linalg.norm(img) + 1e-12)
        txt = self.encode_texts_cached(prompts).astype(np.float32, copy=False)
        return (txt @ img).astype(np.float32)

    @torch.no_grad()
    def score_image_prompts(self, image_path: str, prompts: list[str]) -> np.ndarray:
        """
        Cosine similarities between an image and each prompt (shape: [N]).
        """
        img = self.encode_image(image_path).astype(np.float32, copy=False)
        return self.score_image_prompts_from_emb(img, prompts)