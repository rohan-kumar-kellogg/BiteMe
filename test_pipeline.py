import numpy as np
import pandas as pd

from models.vision import VisionEncoder
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

DATA_DIR = "data"

print("Loading assets...")

encoder = VisionEncoder()

cuisines = pd.read_csv(f"{DATA_DIR}/cuisines.csv")["cuisine"].tolist()
dish_classes = pd.read_csv(f"{DATA_DIR}/dish_classes.csv")["dish_class"].tolist()
ingredients_master = pd.read_csv(f"{DATA_DIR}/ingredients_master.csv")["ingredient"].tolist()
taste_attrs_df = pd.read_csv(f"{DATA_DIR}/taste_attributes.csv")
affinity_df = pd.read_csv(f"{DATA_DIR}/semantic_affinities.csv")

archetypes_df = pd.read_csv(f"{DATA_DIR}/user_archetypes.csv")
user_embeddings = np.load(f"{DATA_DIR}/user_embeddings.npy")

restaurants_df = pd.read_csv(f"{DATA_DIR}/restaurants.csv")
restaurant_embeddings = np.load(f"{DATA_DIR}/restaurant_embeddings.npy")

print("Assets loaded successfully")

# -------- simulate a dish --------

sample = list(Path("images").rglob("*.jpg"))[:1]
img_path = str(sample[0]) if sample else "example_food.jpg"

print("Encoding image...")
img_emb = encoder.encode_image(img_path)

print("Image embedding shape:", img_emb.shape)

# attribute vector (CLIP prompt scoring)
attr_vec = []
for row in taste_attrs_df.itertuples(index=False):
    pos_prompts = [p.strip() for p in str(row.positive_prompt).split("|") if p.strip()]
    neg_prompts = [p.strip() for p in str(row.negative_prompt).split("|") if p.strip()]
    pos = encoder.score_image_prompts(img_path, pos_prompts).astype(np.float32)
    neg = encoder.score_image_prompts(img_path, neg_prompts).astype(np.float32)
    gap = float(np.mean(pos) - np.mean(neg))
    attr_vec.append(float(1.0 / (1.0 + float(np.exp(-gap / 0.07)))))
attr_vec = np.asarray(attr_vec, dtype=np.float32)

# semantic affinity vector
aff_vec = []
for row in affinity_df.itertuples(index=False):
    pos_prompts = [p.strip() for p in str(row.positive_prompt).split("|") if p.strip()]
    neg_prompts = [p.strip() for p in str(row.negative_prompt).split("|") if p.strip()]
    pos = encoder.score_image_prompts(img_path, pos_prompts).astype(np.float32)
    neg = encoder.score_image_prompts(img_path, neg_prompts).astype(np.float32)
    gap = float(np.mean(pos) - np.mean(neg))
    aff_vec.append(float(1.0 / (1.0 + float(np.exp(-gap / 0.07)))))
aff_vec = np.asarray(aff_vec, dtype=np.float32)

# build dish vector (pure CLIP image embedding)
dish_vec = img_emb / np.linalg.norm(img_emb)

print("Dish vector shape:", dish_vec.shape)

# simulate user profile
user_vec = dish_vec

# -------- find user matches --------

print("Finding matches...")

sims = cosine_similarity([user_vec], user_embeddings)[0]
top_idx = np.argsort(-sims)[:5]

for i in top_idx:
    archetype = archetypes_df.iloc[i]["archetype"]
    print(f"Match user {i} | archetype: {archetype} | similarity: {sims[i]:.3f}")

# -------- restaurant ranking --------

print("\nTop restaurants:")

sims = cosine_similarity([user_vec], restaurant_embeddings)[0]
top_idx = np.argsort(-sims)[:5]

for i in top_idx:
    r = restaurants_df.iloc[i]
    print(f"{r['name']} | {r['cuisine']} | score {sims[i]:.3f}")

print("\nPipeline works.")