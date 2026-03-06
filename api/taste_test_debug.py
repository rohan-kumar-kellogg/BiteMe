from taste_profile import debug_applied_taste_traits

tests = [
    "tiramisu",
    "brownie",
    "buffalo wings",
    "mapo tofu",
    "ramen",
    "fried chicken",
    "caesar salad",
    "ceviche",
    "mac and cheese",
    "burger and fries",
    "sushi",
    "steak",
]

for t in tests:
    print("\n", t)
    print(debug_applied_taste_traits(t))