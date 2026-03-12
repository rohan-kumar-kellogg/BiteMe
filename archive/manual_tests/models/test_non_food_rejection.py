from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import models.retrieval as retrieval
from api.profile_logic import empty_profile, update_profile_from_prediction


class NonFoodRejectionTests(unittest.TestCase):
    def test_non_food_gate_rejects_obvious_non_food_like_grass(self):
        # Represents a "grass/plant" style frame: weak food gate, ambiguous nearest-food match.
        reject = retrieval._should_reject_as_not_food(
            predicted_score=0.91,
            second_score=0.89,
            confidence_threshold=0.86,
            food_gate_score=0.24,
        )
        self.assertTrue(reject)

    def test_non_food_gate_rejects_blank_wall_like_input(self):
        reject = retrieval._should_reject_as_not_food(
            predicted_score=0.90,
            second_score=0.88,
            confidence_threshold=0.86,
            food_gate_score=0.32,
        )
        self.assertTrue(reject)

    def test_non_food_gate_keeps_clear_food_predictions(self):
        # Represents clear food classes like pizza/ramen/dessert.
        reject = retrieval._should_reject_as_not_food(
            predicted_score=0.94,
            second_score=0.81,
            confidence_threshold=0.86,
            food_gate_score=0.74,
        )
        self.assertFalse(reject)

    def test_predict_with_confidence_returns_not_food_when_gate_rejects(self):
        orig_predict_dish = retrieval.predict_dish
        orig_food_gate = retrieval._food_gate_score
        try:
            retrieval.predict_dish = lambda *args, **kwargs: [  # type: ignore[assignment]
                {
                    "dish_label": "tacos",
                    "dish_class": "tacos",
                    "cuisine": "mexican",
                    "protein_type": "",
                    "final_score": 0.91,
                    "combined_retrieval": 0.90,
                    "mlp_blend_score": 0.90,
                    "pair_agreement": 0.0,
                    "image_path": "dummy.jpg",
                    "query_embedding": [0.1, 0.2, 0.3],
                },
                {
                    "dish_label": "ramen",
                    "dish_class": "ramen",
                    "cuisine": "japanese",
                    "protein_type": "",
                    "final_score": 0.89,
                    "combined_retrieval": 0.88,
                    "mlp_blend_score": 0.88,
                    "pair_agreement": 0.0,
                    "image_path": "dummy2.jpg",
                    "query_embedding": [0.1, 0.2, 0.3],
                },
            ]
            retrieval._food_gate_score = lambda encoder, emb: (0.22, -0.12, 0.05)  # type: ignore[assignment]

            out = retrieval.predict_dish_with_confidence(
                image_path="dummy.jpg",
                dishes_df=pd.DataFrame([{"dish_label": "tacos"}]),
                dish_vectors=np.zeros((1, 3), dtype=np.float32),
                encoder=object(),  # only used as a passthrough to mocked food gate
                confidence_threshold=0.86,
            )
            self.assertTrue(out["abstained"])
            self.assertTrue(out["rejected_as_not_food"])
            self.assertEqual(out["predicted_label"], "not_food")
            self.assertEqual(out["top3_candidates"], [])
        finally:
            retrieval.predict_dish = orig_predict_dish  # type: ignore[assignment]
            retrieval._food_gate_score = orig_food_gate  # type: ignore[assignment]

    def test_abstained_not_food_does_not_mutate_taste_memory(self):
        profile = empty_profile()
        updated = update_profile_from_prediction(
            profile,
            {
                "predicted_label": "not_food",
                "abstained": True,
                "top3_candidates": [
                    {"dish_label": "tacos", "final_score": 0.91, "cuisine": "mexican"},
                    {"dish_label": "ramen", "final_score": 0.89, "cuisine": "japanese"},
                ],
            },
        )
        self.assertEqual(updated.get("favorite_dishes", {}), {})
        self.assertEqual(updated.get("favorite_cuisines", {}), {})
        self.assertEqual(int(updated.get("upload_count", 0)), 1)


if __name__ == "__main__":
    unittest.main()

