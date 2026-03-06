import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"

def load_data():
    user_embeddings = np.load(f"{DATA_DIR}/user_embeddings.npy")
    pair_features = np.load(f"{DATA_DIR}/pair_features.npy")
    pairs_df = pd.read_csv(f"{DATA_DIR}/compatibility_pairs.csv")
    archetypes_df = pd.read_csv(f"{DATA_DIR}/user_archetypes.csv")
    return user_embeddings, pair_features, pairs_df, archetypes_df

def train_compatibility_model(pair_features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        pair_features, labels, test_size=0.2, random_state=42
    )

    # Ridge is usually a bit more stable than plain linear regression
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    return model, r2

def example_match(user_embeddings, archetypes_df, model, userA=0, userB=1):
    pair_vec = np.abs(user_embeddings[userA] - user_embeddings[userB])
    pred_score = float(model.predict([pair_vec])[0])

    sim = float(cosine_similarity([user_embeddings[userA]], [user_embeddings[userB]])[0][0])
    aA = archetypes_df.loc[archetypes_df.user_id == userA, "archetype"].values[0]
    aB = archetypes_df.loc[archetypes_df.user_id == userB, "archetype"].values[0]

    print("\n--- Example Match ---")
    print(f"User {userA}: {aA}")
    print(f"User {userB}: {aB}")
    print(f"Cosine similarity: {sim:.3f}")
    print(f"Predicted satisfaction score: {pred_score:.3f}")

def main():
    user_embeddings, pair_features, pairs_df, archetypes_df = load_data()

    labels = pairs_df["satisfaction_score"].values
    model, r2 = train_compatibility_model(pair_features, labels)

    print(f"📊 Compatibility model R² (test): {r2:.3f}")

    example_match(user_embeddings, archetypes_df, model, userA=0, userB=1)

if __name__ == "__main__":
    main()