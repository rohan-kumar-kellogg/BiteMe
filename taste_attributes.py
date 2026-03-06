from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.vision import VisionEncoder


@dataclass(frozen=True)
class AttributeSpec:
    name: str
    positive_prompts: tuple[str, ...]
    negative_prompts: tuple[str, ...]


# These are intentionally "visual" prompts (what the dish looks like),
# since we're trying to infer attributes directly from photos.
ATTRIBUTE_SPECS: list[AttributeSpec] = [
    AttributeSpec(
        "spicy",
        (
            "a very spicy dish with red chili peppers",
            "spicy food with chili oil and pepper flakes",
            "a bowl of spicy curry with visible chili",
        ),
        (
            "a mild dish, not spicy",
            "gentle flavors, no chili peppers",
            "not spicy food",
        ),
    ),
    AttributeSpec(
        "umami",
        (
            "an umami-rich savory dish, deeply savory",
            "a dish with soy sauce, miso, or mushrooms",
            "savory broth, ramen-like umami",
        ),
        (
            "a bland dish, not savory",
            "not umami, very plain food",
            "watery soup with little flavor",
        ),
    ),
    AttributeSpec(
        "sweet",
        (
            "a sweet dessert, sugary, sweet food",
            "a pastry or cake with icing",
            "sweet syrup or caramel on dessert",
        ),
        (
            "not sweet, savory food",
            "no sugar, not dessert",
            "a savory main course, not sweet",
        ),
    ),
    AttributeSpec(
        "sour",
        (
            "a sour dish, citrus, vinegar, tangy",
            "a dish with lemon wedges, very tangy",
            "pickled or vinegar-forward food",
        ),
        (
            "not sour, not tangy",
            "no citrus, no vinegar",
            "neutral tasting food, not sour",
        ),
    ),
    AttributeSpec(
        "creamy",
        (
            "a creamy rich dish with creamy sauce",
            "cheesy and buttery, rich creamy texture",
            "a dish covered in white cream sauce",
        ),
        (
            "not creamy, dry food",
            "clear broth, not creamy",
            "no sauce, not rich",
        ),
    ),
    AttributeSpec(
        "crispy",
        (
            "a crispy crunchy fried dish, crunchy texture",
            "golden-brown fried food, very crunchy",
            "crispy breaded cutlet with crunch",
        ),
        (
            "not crispy, soft texture",
            "steamed or boiled food, soft",
            "no crunch, not fried",
        ),
    ),
    AttributeSpec(
        "smoky",
        (
            "a smoky grilled dish, charred, barbecue",
            "grilled meat with char marks, smoky",
            "barbecue sauce and smoked flavor",
        ),
        (
            "not smoky, not grilled",
            "no char, not barbecue",
            "steamed food, not smoky",
        ),
    ),
]


def attribute_names() -> list[str]:
    return [a.name for a in ATTRIBUTE_SPECS]


def compute_attribute_scores(
    encoder: VisionEncoder,
    image_path: str,
    *,
    temperature: float = 0.07,
) -> np.ndarray:
    """
    Returns scores in [0, 1] for each attribute in `ATTRIBUTE_SPECS`, in order.
    Uses CLIP image-text similarity with a 2-way softmax for (positive vs negative).
    """
    out: list[float] = []
    for spec in ATTRIBUTE_SPECS:
        pos = encoder.score_image_prompts(image_path, list(spec.positive_prompts)).astype(np.float32)
        neg = encoder.score_image_prompts(image_path, list(spec.negative_prompts)).astype(np.float32)

        # Use a sigmoid on the pos-vs-neg logit gap for better spread than 2-way softmax.
        gap = float(np.mean(pos) - np.mean(neg))
        t = max(float(temperature), 1e-6)
        score = 1.0 / (1.0 + float(np.exp(-gap / t)))
        out.append(float(score))
    return np.asarray(out, dtype=np.float32)

