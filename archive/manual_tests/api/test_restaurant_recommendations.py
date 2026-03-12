from __future__ import annotations

import unittest

from api.restaurant_recommendations import (
    RestaurantProvider,
    debug_restaurant_recommendation,
    get_compatible_restaurants,
)


def _profile(
    *,
    cuisines: dict[str, float],
    dishes: dict[str, float],
    traits: dict[str, float],
    dims: dict[str, float],
    behavior: dict[str, float] | None = None,
) -> dict:
    return {
        "favorite_cuisines": cuisines,
        "favorite_dishes": dishes,
        "favorite_traits": traits,
        "behavior_features": behavior or {},
        "taste_profile": {"dimensions": {k: {"score": v} for k, v in dims.items()}},
    }


class RestaurantRecommendationRankingTests(unittest.TestCase):
    @staticmethod
    def _provider() -> RestaurantProvider:
        class FixtureProvider(RestaurantProvider):
            source_name = "test_fixture"

            _rows = [
                {
                    "id": "rest_001",
                    "name": "Basil & Brick",
                    "address": "123 Main St, Evanston, IL",
                    "zip_code": "60201",
                    "latitude": 42.0548,
                    "longitude": -87.6859,
                    "cuisine_tags": ["italian"],
                    "menu_tags": ["pizza", "pasta", "classic", "dessert", "tiramisu", "cozy"],
                    "trait_tags": ["cozy", "hearty", "classic", "dessert"],
                    "venue_type": "sit_down",
                    "service_type": "full_service",
                    "rating": 4.6,
                    "review_count": 420,
                    "reservation_provider": "opentable",
                    "reservation_url": "https://www.opentable.com/",
                    "website_url": "https://example.com/basil",
                    "phone": "+1-847-555-1001",
                    "source": "test_fixture",
                },
                {
                    "id": "rest_002",
                    "name": "Sweet Theory Patisserie",
                    "address": "45 Oak Ave, Evanston, IL",
                    "zip_code": "60201",
                    "latitude": 42.0532,
                    "longitude": -87.6892,
                    "cuisine_tags": ["french", "italian"],
                    "menu_tags": ["dessert", "pastry", "bakery", "gelato", "cannoli"],
                    "trait_tags": ["dessert", "cafe", "classic"],
                    "venue_type": "dessert",
                    "service_type": "counter_service",
                    "rating": 4.7,
                    "review_count": 310,
                    "reservation_provider": "",
                    "reservation_url": "",
                    "website_url": "https://example.com/sweet",
                    "phone": "+1-847-555-1002",
                    "source": "test_fixture",
                },
                {
                    "id": "rest_004",
                    "name": "Midnight Noodle House",
                    "address": "88 Lake St, Evanston, IL",
                    "zip_code": "60201",
                    "latitude": 42.0551,
                    "longitude": -87.6838,
                    "cuisine_tags": ["japanese"],
                    "menu_tags": ["ramen", "noodles", "broth", "umami", "late_night"],
                    "trait_tags": ["hearty", "umami", "casual"],
                    "venue_type": "sit_down",
                    "service_type": "full_service",
                    "rating": 4.5,
                    "review_count": 890,
                    "reservation_provider": "resy",
                    "reservation_url": "https://resy.com/",
                    "website_url": "https://example.com/noodle",
                    "phone": "+1-847-555-1004",
                    "source": "test_fixture",
                },
                {
                    "id": "rest_005",
                    "name": "Saffron Current",
                    "address": "205 State St, Evanston, IL",
                    "zip_code": "60201",
                    "latitude": 42.0519,
                    "longitude": -87.6911,
                    "cuisine_tags": ["indian"],
                    "menu_tags": ["spicy", "chili_forward", "curry", "global", "hearty"],
                    "trait_tags": ["spicy", "cozy", "classic"],
                    "venue_type": "sit_down",
                    "service_type": "full_service",
                    "rating": 4.6,
                    "review_count": 710,
                    "reservation_provider": "opentable",
                    "reservation_url": "https://www.opentable.com/",
                    "website_url": "https://example.com/saffron",
                    "phone": "+1-847-555-1005",
                    "source": "test_fixture",
                },
                {
                    "id": "rest_007",
                    "name": "Seoul Ember",
                    "address": "910 Grand Ave, Chicago, IL",
                    "zip_code": "60611",
                    "latitude": 41.9009,
                    "longitude": -87.6230,
                    "cuisine_tags": ["korean"],
                    "menu_tags": ["spicy", "grill", "protein_forward", "late_night", "chef_driven"],
                    "trait_tags": ["spicy", "adventurous", "hearty"],
                    "venue_type": "sit_down",
                    "service_type": "full_service",
                    "rating": 4.7,
                    "review_count": 510,
                    "reservation_provider": "resy",
                    "reservation_url": "https://resy.com/",
                    "website_url": "https://example.com/seoul",
                    "phone": "+1-312-555-1007",
                    "source": "test_fixture",
                },
                {
                    "id": "rest_008",
                    "name": "Atlas Omakase Bar",
                    "address": "1010 Pine St, Chicago, IL",
                    "zip_code": "60614",
                    "latitude": 41.9237,
                    "longitude": -87.6519,
                    "cuisine_tags": ["japanese"],
                    "menu_tags": ["omakase", "tasting_menu", "chef_driven", "global", "sushi"],
                    "trait_tags": ["adventurous", "premium", "precise"],
                    "venue_type": "sit_down",
                    "service_type": "full_service",
                    "rating": 4.8,
                    "review_count": 340,
                    "reservation_provider": "resy",
                    "reservation_url": "https://resy.com/",
                    "website_url": "https://example.com/atlas",
                    "phone": "+1-312-555-1008",
                    "source": "test_fixture",
                },
                {
                    "id": "rest_009",
                    "name": "Green Thread Kitchen",
                    "address": "222 Elm St, Evanston, IL",
                    "zip_code": "60201",
                    "latitude": 42.0526,
                    "longitude": -87.6880,
                    "cuisine_tags": ["mediterranean"],
                    "menu_tags": ["healthy", "salad", "clean_eating", "vegetarian_friendly", "light"],
                    "trait_tags": ["fresh", "light", "healthy"],
                    "venue_type": "quick_bite",
                    "service_type": "quick_service",
                    "rating": 4.3,
                    "review_count": 260,
                    "reservation_provider": "",
                    "reservation_url": "",
                    "website_url": "https://example.com/green",
                    "phone": "+1-847-555-1009",
                    "source": "test_fixture",
                },
                {
                    "id": "rest_003",
                    "name": "Fireline Taqueria",
                    "address": "77 Maple St, Evanston, IL",
                    "zip_code": "60202",
                    "latitude": 42.0318,
                    "longitude": -87.6904,
                    "cuisine_tags": ["mexican"],
                    "menu_tags": ["spicy", "chili_forward", "tacos", "casual", "hearty"],
                    "trait_tags": ["casual", "spicy", "comfort"],
                    "venue_type": "quick_bite",
                    "service_type": "quick_service",
                    "rating": 4.4,
                    "review_count": 560,
                    "reservation_provider": "",
                    "reservation_url": "",
                    "website_url": "https://example.com/fireline",
                    "phone": "+1-847-555-1003",
                    "source": "test_fixture",
                },
                {
                    "id": "rest_006",
                    "name": "Lime & Tide Cevicheria",
                    "address": "17 Cedar St, Evanston, IL",
                    "zip_code": "60202",
                    "latitude": 42.0302,
                    "longitude": -87.6871,
                    "cuisine_tags": ["mexican", "middle_eastern"],
                    "menu_tags": ["ceviche", "seafood", "light", "fresh", "clean_eating"],
                    "trait_tags": ["healthy", "light", "fresh"],
                    "venue_type": "sit_down",
                    "service_type": "full_service",
                    "rating": 4.5,
                    "review_count": 290,
                    "reservation_provider": "resy",
                    "reservation_url": "https://resy.com/",
                    "website_url": "https://example.com/lime",
                    "phone": "+1-847-555-1006",
                    "source": "test_fixture",
                },
                {
                    "id": "rest_011",
                    "name": "Noodle Circuit",
                    "address": "540 Ashland Ave, Chicago, IL",
                    "zip_code": "60622",
                    "latitude": 41.9033,
                    "longitude": -87.6747,
                    "cuisine_tags": ["thai", "chinese"],
                    "menu_tags": ["noodles", "spicy", "szechuan", "thai", "global"],
                    "trait_tags": ["adventurous", "spicy", "casual"],
                    "venue_type": "quick_bite",
                    "service_type": "quick_service",
                    "rating": 4.4,
                    "review_count": 450,
                    "reservation_provider": "",
                    "reservation_url": "",
                    "website_url": "https://example.com/circuit",
                    "phone": "+1-312-555-1011",
                    "source": "test_fixture",
                },
                {
                    "id": "rest_012",
                    "name": "Cafe Parlor 27",
                    "address": "315 Grove St, Evanston, IL",
                    "zip_code": "60201",
                    "latitude": 42.0545,
                    "longitude": -87.6901,
                    "cuisine_tags": ["french", "american"],
                    "menu_tags": ["cafe", "brunch", "pastry", "bakery", "light"],
                    "trait_tags": ["cafe", "fresh", "dessert"],
                    "venue_type": "cafe",
                    "service_type": "counter_service",
                    "rating": 4.5,
                    "review_count": 390,
                    "reservation_provider": "",
                    "reservation_url": "",
                    "website_url": "https://example.com/parlor",
                    "phone": "+1-847-555-1012",
                    "source": "test_fixture",
                },
            ]

            def get_restaurants_by_zip(self, zip_code: str, limit: int = 200):
                z = "".join(ch for ch in str(zip_code) if ch.isdigit())[:5]
                rows = [r for r in self._rows if (not z or r["zip_code"] == z)]
                return rows[:limit]

        return FixtureProvider()

    # Case 1: user in 60201 with strong dessert + strong italian preference.
    def test_dessert_italian_profile_prefers_italian_dessert_nearby(self):
        p = _profile(
            cuisines={"italian": 4.0, "french": 1.5},
            dishes={"tiramisu": 2.0, "gelato": 1.8, "cannoli": 1.6},
            traits={"dessert-leaning": 2.0},
            dims={"dessert_affinity": 0.92, "sweet_leaning": 0.86, "freshness_preference": 0.42},
        )
        rows = get_compatible_restaurants(p, zip_code="60201", limit=5, provider=self._provider())
        top_names = [r["restaurant"]["name"] for r in rows[:3]]
        self.assertTrue(any(x in top_names for x in ["Basil & Brick", "Sweet Theory Patisserie"]))

    # Case 2: user in 60201 with spicy preference + mexican/thai history.
    def test_spicy_mexican_thai_profile_prefers_heat_and_cuisine_fit(self):
        p = _profile(
            cuisines={"mexican": 2.8, "thai": 2.4},
            dishes={"mapo tofu": 1.5, "buffalo wings": 1.4},
            traits={"spice-forward": 2.3},
            dims={"spicy_leaning": 0.9, "dessert_affinity": 0.3, "freshness_preference": 0.45},
            behavior={"adventurousness": 0.7},
        )
        rows = get_compatible_restaurants(p, zip_code="60201", limit=5, provider=self._provider())
        top_names = [r["restaurant"]["name"] for r in rows[:3]]
        self.assertTrue(any(x in top_names for x in ["Fireline Taqueria", "Saffron Current", "Noodle Circuit"]))

    # Case 3: weak cuisine signals, strong bakery/cafe behavior.
    def test_bakery_behavior_without_strong_cuisine_still_finds_cafe_dessert(self):
        p = _profile(
            cuisines={"american": 1.0},
            dishes={"pastry": 1.8, "brownie": 1.2, "croissant": 1.2},
            traits={"dessert-leaning": 2.0},
            dims={"dessert_affinity": 0.84, "sweet_leaning": 0.82, "freshness_preference": 0.55},
        )
        rows = get_compatible_restaurants(p, zip_code="60201", limit=5, provider=self._provider())
        top = rows[0]["restaurant"]["name"]
        self.assertIn(top, {"Sweet Theory Patisserie", "Cafe Parlor 27"})

    # Case 4: adventurous eater with japanese/korean dish history.
    def test_adventurous_japanese_korean_prefers_omakase_or_korean(self):
        p = _profile(
            cuisines={"japanese": 2.2, "korean": 1.9},
            dishes={"sushi": 1.6, "ramen": 1.4},
            traits={"spice-forward": 0.8},
            dims={"spicy_leaning": 0.62, "umami_leaning": 0.82, "freshness_preference": 0.58},
            behavior={"adventurousness": 0.84},
        )
        rows = get_compatible_restaurants(p, zip_code="60614", limit=5, provider=self._provider())
        top_names = [r["restaurant"]["name"] for r in rows[:3]]
        self.assertTrue(any(x in top_names for x in ["Atlas Omakase Bar", "Seoul Ember", "Midnight Noodle House"]))

    # Case 5: healthy leaning with mediterranean/salad/bowl history.
    def test_healthy_mediterranean_profile_prefers_fresh_light_places(self):
        p = _profile(
            cuisines={"mediterranean": 2.4},
            dishes={"salad bowl": 1.8, "ceviche": 1.1},
            traits={"plant-forward": 1.9},
            dims={"freshness_preference": 0.9, "richness_preference": 0.35, "dessert_affinity": 0.22},
        )
        rows = get_compatible_restaurants(p, zip_code="60201", limit=5, provider=self._provider())
        top_names = [r["restaurant"]["name"] for r in rows[:3]]
        self.assertTrue(any(x in top_names for x in ["Green Thread Kitchen", "Lime & Tide Cevicheria"]))

    # Case 6: location practicality should matter (same profile, far zip).
    def test_location_shift_changes_top_results(self):
        p = _profile(
            cuisines={"japanese": 2.0},
            dishes={"ramen": 1.8},
            traits={"comfort-food": 1.2},
            dims={"umami_leaning": 0.8, "richness_preference": 0.72},
        )
        near_rows = get_compatible_restaurants(p, zip_code="60201", limit=3, provider=self._provider())
        far_rows = get_compatible_restaurants(p, zip_code="60622", limit=3, provider=self._provider())
        # Location should materially influence ranking, but not erase strong taste fit.
        # Assert that the top-3 composition shifts when ZIP changes.
        near_ids = [r["restaurant"]["id"] for r in near_rows]
        far_ids = [r["restaurant"]["id"] for r in far_rows]
        self.assertNotEqual(near_ids, far_ids)

    # One-off dessert upload should not overpower strong italian cuisine preference.
    def test_one_off_dessert_does_not_overpower_strong_italian(self):
        p = _profile(
            cuisines={"italian": 6.0, "french": 0.4},
            dishes={"tiramisu": 0.55},
            traits={"dessert-leaning": 0.4},
            dims={"dessert_affinity": 0.64, "sweet_leaning": 0.62, "freshness_preference": 0.5},
        )
        rows = get_compatible_restaurants(p, zip_code="60201", limit=5, provider=self._provider())
        self.assertEqual(rows[0]["restaurant"]["name"], "Basil & Brick")

    # Repeated dessert behavior should allow dessert-forward places to lead.
    def test_repeated_dessert_behavior_can_elevate_dessert_shop(self):
        p = _profile(
            cuisines={"italian": 1.6, "french": 1.3},
            dishes={"tiramisu": 2.2, "gelato": 2.0, "cannoli": 1.9, "pastry": 1.5},
            traits={"dessert-leaning": 2.8},
            dims={"dessert_affinity": 0.93, "sweet_leaning": 0.9, "freshness_preference": 0.42},
        )
        rows = get_compatible_restaurants(p, zip_code="60201", limit=5, provider=self._provider())
        top_names = [r["restaurant"]["name"] for r in rows[:2]]
        self.assertIn("Sweet Theory Patisserie", top_names)

    # Repeated ramen/sushi behavior should rank japanese-focused venues highly.
    def test_repeated_ramen_sushi_behavior_prefers_japanese_spots(self):
        p = _profile(
            cuisines={"japanese": 3.5, "korean": 0.9},
            dishes={"ramen": 2.5, "sushi": 2.3, "udon": 1.4},
            traits={"protein-forward": 1.0},
            dims={"umami_leaning": 0.88, "freshness_preference": 0.58, "spicy_leaning": 0.52},
            behavior={"adventurousness": 0.72},
        )
        rows = get_compatible_restaurants(p, zip_code="60201", limit=5, provider=self._provider())
        top_names = [r["restaurant"]["name"] for r in rows[:3]]
        self.assertTrue(any(x in top_names for x in ["Midnight Noodle House", "Atlas Omakase Bar"]))

    # Mixed italian + dessert profile should keep both classes competitive.
    def test_mixed_italian_and_dessert_profile_keeps_both_competitive(self):
        p = _profile(
            cuisines={"italian": 3.2, "french": 1.0},
            dishes={"pasta": 2.0, "tiramisu": 1.8, "gelato": 1.4},
            traits={"dessert-leaning": 1.8, "comfort-food": 1.2},
            dims={"dessert_affinity": 0.82, "sweet_leaning": 0.76, "richness_preference": 0.72},
        )
        rows = get_compatible_restaurants(p, zip_code="60201", limit=5, provider=self._provider())
        top_names = [r["restaurant"]["name"] for r in rows[:3]]
        self.assertIn("Basil & Brick", top_names)
        self.assertIn("Sweet Theory Patisserie", top_names)

    # Optional context should nudge ranking without replacing base taste matching.
    def test_context_dessert_nudges_results(self):
        p = _profile(
            cuisines={"italian": 2.2, "french": 1.2},
            dishes={"pasta": 1.6, "tiramisu": 1.5},
            traits={"dessert-leaning": 1.5},
            dims={"dessert_affinity": 0.79, "sweet_leaning": 0.71, "richness_preference": 0.69},
        )
        base_rows = get_compatible_restaurants(p, zip_code="60201", limit=3, provider=self._provider())
        dessert_rows = get_compatible_restaurants(p, zip_code="60201", context="dessert", limit=3, provider=self._provider())
        base_sweet = next(r for r in base_rows if r["restaurant"]["id"] == "rest_002")
        ctx_sweet = next(r for r in dessert_rows if r["restaurant"]["id"] == "rest_002")
        self.assertGreaterEqual(ctx_sweet["compatibility_score"], base_sweet["compatibility_score"])

    # Pizza-forward Evanston profile should rank the local pizza venue highly with intuitive score.
    def test_union_pizzeria_ranks_high_for_pizza_evanston_profile(self):
        p = _profile(
            cuisines={"italian": 3.4, "pizza": 2.2},
            dishes={"pizza margherita": 2.4, "pepperoni pizza": 1.8, "tiramisu": 1.2},
            traits={"comfort-food": 1.6},
            dims={"dessert_affinity": 0.62, "sweet_leaning": 0.56, "richness_preference": 0.68},
        )
        rows = get_compatible_restaurants(p, zip_code="60201", limit=5, provider=self._provider())
        union = next(r for r in rows if r["restaurant"]["name"] == "Basil & Brick")
        self.assertGreaterEqual(union["compatibility_score"], 75.0)

    # Local obvious fit should score materially above weak distant fit.
    def test_local_obvious_fit_scores_higher_than_weak_distant_fit(self):
        p = _profile(
            cuisines={"italian": 3.0},
            dishes={"pizza margherita": 2.0, "tiramisu": 1.2},
            traits={"comfort-food": 1.2},
            dims={"dessert_affinity": 0.62, "sweet_leaning": 0.57, "richness_preference": 0.65},
        )
        rows = get_compatible_restaurants(p, zip_code="60201", limit=5, provider=self._provider())
        best = rows[0]
        weak = rows[-1]
        self.assertGreater(best["compatibility_score"], weak["compatibility_score"] + 15.0)

    def test_debug_single_recommendation_has_expected_fields(self):
        p = _profile(
            cuisines={"italian": 3.2, "pizza": 2.0},
            dishes={"pizza margherita": 2.1, "tiramisu": 1.1},
            traits={"comfort-food": 1.2},
            dims={"dessert_affinity": 0.64, "sweet_leaning": 0.56, "richness_preference": 0.67},
        )
        dbg = debug_restaurant_recommendation(
            p,
            "60201",
            restaurant_query="basil",
            provider=self._provider(),
        )
        self.assertIsNotNone(dbg)
        assert dbg is not None
        self.assertIn("matched_cuisines", dbg["debug"])
        self.assertIn("matched_dishes_menu_tags", dbg["debug"])
        self.assertIn("matched_traits", dbg["debug"])
        self.assertIn("location", dbg["debug"])
        self.assertIn("raw_score", dbg["debug"])
        self.assertIn("final_displayed_percent", dbg["debug"])

    def test_repeated_direct_dish_uploads_improve_direct_venue_match(self):
        base = _profile(
            cuisines={"italian": 2.2},
            dishes={"italian dinner": 1.0, "comfort meal": 0.6},
            traits={"comfort-food": 0.6},
            dims={"dessert_affinity": 0.60, "sweet_leaning": 0.54, "richness_preference": 0.63},
        )
        repeated = _profile(
            cuisines={"italian": 3.6, "pizza": 2.1},
            dishes={"pizza margherita": 2.3, "pepperoni pizza": 1.9, "tiramisu": 0.9},
            traits={"comfort-food": 1.4},
            dims={"dessert_affinity": 0.61, "sweet_leaning": 0.55, "richness_preference": 0.69},
        )
        base_rows = get_compatible_restaurants(base, zip_code="60201", limit=5, provider=self._provider())
        rep_rows = get_compatible_restaurants(repeated, zip_code="60201", limit=5, provider=self._provider())
        base_basil = next(r for r in base_rows if r["restaurant"]["name"] == "Basil & Brick")
        rep_basil = next(r for r in rep_rows if r["restaurant"]["name"] == "Basil & Brick")
        self.assertGreater(
            rep_basil["compatibility_score"],
            base_basil["compatibility_score"],
        )


if __name__ == "__main__":
    unittest.main()
