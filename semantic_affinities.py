from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.vision import VisionEncoder


@dataclass(frozen=True)
class AffinitySpec:
    name: str
    positive_prompts: tuple[str, ...]
    negative_prompts: tuple[str, ...]


# These are broad "style" affinities that can fire even when dish class is unseen.
AFFINITY_SPECS: list[AffinitySpec] = [
    AffinitySpec(
        "peruvian",
        (
            "peruvian cuisine, ceviche with red onion and lime",
            "peruvian food plating, andean or coastal peruvian dish",
            "leche de tigre style seafood with citrus",
        ),
        (
            "not peruvian cuisine",
            "generic western dish, no peruvian cues",
            "plain sandwich or burger, not latin seafood",
        ),
    ),
    AffinitySpec(
        "japanese",
        (
            "japanese cuisine, sushi or ramen presentation",
            "japanese dish with clean minimal plating",
            "soy-based japanese food style",
        ),
        (
            "not japanese cuisine",
            "western fast food style",
            "no japanese food cues",
        ),
    ),
    AffinitySpec(
        "italian",
        (
            "italian cuisine, pasta pizza risotto style",
            "italian dish with tomato and basil presentation",
            "rustic italian plating",
        ),
        (
            "not italian cuisine",
            "no pasta pizza or risotto cues",
            "non-italian food style",
        ),
    ),
    AffinitySpec(
        "mexican",
        (
            "mexican cuisine, tacos salsa lime cilantro",
            "mexican food with chili and corn tortilla",
            "street-style mexican dish plating",
        ),
        (
            "not mexican cuisine",
            "no tortilla or salsa cues",
            "non-mexican food style",
        ),
    ),
    AffinitySpec(
        "seafood_focused",
        (
            "seafood dish with fish or shellfish",
            "plated fish fillet or shrimp dish",
            "ceviche-like raw fish with citrus",
        ),
        (
            "meat-heavy dish without seafood",
            "beef or chicken focused plate",
            "no fish or shellfish visible",
        ),
    ),
    AffinitySpec(
        "acidic_bright",
        (
            "acidic bright dish with citrus and vinegar notes",
            "fresh lime or lemon dressed seafood salad",
            "tangy citrus-forward plated food",
        ),
        (
            "rich heavy creamy dish, not acidic",
            "bland or neutral tasting appearance",
            "no citrus or tangy cues",
        ),
    ),
]


def affinity_names() -> list[str]:
    return [a.name for a in AFFINITY_SPECS]


def compute_affinity_scores(
    encoder: VisionEncoder,
    image_path: str,
    *,
    temperature: float = 0.07,
) -> np.ndarray:
    out: list[float] = []
    for spec in AFFINITY_SPECS:
        pos = encoder.score_image_prompts(image_path, list(spec.positive_prompts)).astype(np.float32)
        neg = encoder.score_image_prompts(image_path, list(spec.negative_prompts)).astype(np.float32)
        gap = float(np.mean(pos) - np.mean(neg))
        t = max(float(temperature), 1e-6)
        out.append(float(1.0 / (1.0 + float(np.exp(-gap / t)))))
    return np.asarray(out, dtype=np.float32)

