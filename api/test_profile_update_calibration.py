from __future__ import annotations

import unittest

from api.profile_logic import empty_profile, update_profile_from_prediction
from api.profile_logic import update_profile_from_recommendation_click


def _prediction(label: str, *, cuisine: str = "Unknown", score: float = 0.92) -> dict:
    return {
        "predicted_label": label,
        "predicted_score": score,
        "top3_candidates": [
            {
                "dish_label": label,
                "dish_class": label,
                "cuisine": cuisine,
                "final_score": score,
            }
        ],
    }


class TasteProfileCalibrationTests(unittest.TestCase):
    def test_recommendation_click_updates_fingerprint_dimensions(self):
        profile = empty_profile()

        profile = update_profile_from_recommendation_click(
            profile,
            dish_label="spaghetti",
            cuisine="italian",
            signal_weight=0.24,
            record_event=True,
        )
        after_carb = float(profile["taste_profile"]["dimensions"]["carb_forward"]["score"])
        after_comfort = float(profile["taste_profile"]["dimensions"]["comfort_food_tendency"]["score"])
        dims = profile["taste_profile"]["dimensions"]
        shifted = [
            abs(float(v.get("score", 0.5)) - 0.5)
            for v in dims.values()
        ]
        self.assertGreater(after_carb, 0.5)
        self.assertGreater(after_comfort, 0.5)
        self.assertTrue(any(delta >= 0.03 for delta in shifted))
        self.assertGreater(float(profile.get("favorite_cuisines", {}).get("italian", 0.0)), 0.0)
        self.assertGreater(float(profile.get("favorite_dishes", {}).get("spaghetti", 0.0)), 0.0)

    def test_dessert_then_non_dessert_preserves_dessert_affinity_above_baseline(self):
        profile = empty_profile()
        baseline = float(profile["taste_profile"]["dimensions"]["dessert_affinity"]["score"])

        profile = update_profile_from_prediction(profile, _prediction("tiramisu", cuisine="italian", score=0.96))
        after_dessert = float(profile["taste_profile"]["dimensions"]["dessert_affinity"]["score"])
        self.assertGreater(after_dessert, baseline)

        profile = update_profile_from_prediction(profile, _prediction("pizza margherita", cuisine="italian", score=0.95))
        after_non_dessert = float(profile["taste_profile"]["dimensions"]["dessert_affinity"]["score"])
        self.assertGreater(after_non_dessert, baseline)

    def test_second_dessert_upload_increases_dessert_signals_noticeably(self):
        profile = empty_profile()
        profile = update_profile_from_prediction(profile, _prediction("ramen", cuisine="japanese", score=0.93))
        before_dessert = float(profile["taste_profile"]["dimensions"]["dessert_affinity"]["score"])
        before_sweet = float(profile["taste_profile"]["dimensions"]["sweet_leaning"]["score"])

        profile = update_profile_from_prediction(profile, _prediction("tiramisu", cuisine="italian", score=0.96))
        after_dessert = float(profile["taste_profile"]["dimensions"]["dessert_affinity"]["score"])
        after_sweet = float(profile["taste_profile"]["dimensions"]["sweet_leaning"]["score"])

        self.assertGreater(after_dessert - before_dessert, 0.08)
        self.assertGreater(after_sweet - before_sweet, 0.08)

    def test_repeated_dessert_uploads_compound(self):
        profile = empty_profile()
        series = ["tiramisu", "brownie", "cheesecake"]
        dessert_scores: list[float] = []
        for dish in series:
            profile = update_profile_from_prediction(profile, _prediction(dish, cuisine="dessert", score=0.95))
            dessert_scores.append(float(profile["taste_profile"]["dimensions"]["dessert_affinity"]["score"]))
        self.assertGreater(dessert_scores[1], dessert_scores[0])
        self.assertGreater(dessert_scores[2], dessert_scores[1])
        self.assertGreater(dessert_scores[2], 0.72)

    def test_pizza_upload_reinforces_italian_and_pizza_signals(self):
        profile = empty_profile()
        profile = update_profile_from_prediction(profile, _prediction("pizza margherita", cuisine="Unknown", score=0.94))
        fav_cuisines = profile.get("favorite_cuisines", {})
        fav_dishes = profile.get("favorite_dishes", {})
        fav_traits = profile.get("favorite_traits", {})

        self.assertGreater(float(fav_cuisines.get("italian", 0.0)), 0.70)
        self.assertTrue(any("pizza" in k.lower() for k in fav_dishes.keys()))
        self.assertGreater(float(fav_traits.get("comfort-food", 0.0)), 0.30)


if __name__ == "__main__":
    unittest.main()

