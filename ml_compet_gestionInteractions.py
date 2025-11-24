

# ml_compet_gestionInteractions.py

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

ITEMS_FILE = "resultats_enrichie/books_enriched_BASE.xlsx"
ITEMS_SHEET = "Sheet1"

INTERACTIONS_FILE = "kaggle_data/interactions_train.csv"
SAMPLE_SUB_FILE = "kaggle_data/sample_submission.csv"

TOP_K = 10

# =========================================================
# 1. LOAD DATA
# =========================================================

def load_data():
    print(f"📘 Items : {ITEMS_FILE}")
    items_df = pd.read_excel(ITEMS_FILE, sheet_name=ITEMS_SHEET)

    print(f"📘 Interactions : {INTERACTIONS_FILE}")
    interactions = pd.read_csv(INTERACTIONS_FILE)

    print(f"📘 Sample Submission : {SAMPLE_SUB_FILE}")
    sample_sub = pd.read_csv(SAMPLE_SUB_FILE)

    print("Items :", items_df.shape)
    print("Interactions :", interactions.shape)
    return items_df, interactions, sample_sub


# =========================================================
# 2. SPLIT — Leave-One-Out par utilisateur
# =========================================================

def leave_one_out_split(interactions: pd.DataFrame):
    # Tri temporel
    df = interactions.sort_values(["u", "t"]).copy()

    # IMPORTANT : 1 seule interaction max par (u, i)
    before = df.shape[0]
    df = df.drop_duplicates(subset=["u", "i"])
    after = df.shape[0]
    print(f"🧹 Drop doublons (u,i) : {before} -> {after}")

    # Dernière interaction de chaque user = Test
    test = df.groupby("u").tail(1)
    train = df.drop(test.index)

    print(f"Train: {train.shape} Test: {test.shape}")
    return train, test


# =========================================================
# 3. MATRICE USER–ITEM
# =========================================================

def build_R(train_df: pd.DataFrame):
    users = sorted(train_df["u"].unique())
    items = sorted(train_df["i"].unique())

    user_to_idx = {u: idx for idx, u in enumerate(users)}
    item_to_idx = {it: idx for idx, it in enumerate(items)}
    idx_to_item = {idx: it for it, idx in item_to_idx.items()}

    # Debug
    print(f"🔢 Nb users uniques: {len(users)}")
    print(f"🔢 Nb items uniques: {len(items)}")

    rows = train_df["u"].map(user_to_idx).values
    cols = train_df["i"].map(item_to_idx).values
    data = np.ones(len(train_df), dtype=np.float32)

    R = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    print(f"📊 R = {R.shape}, nnz={R.nnz}")

    # Stats interactions par user
    inter_per_user = np.asarray(R.sum(axis=1)).ravel()
    print("📈 Interactions/user : min =", inter_per_user.min(),
          "max =", inter_per_user.max(),
          "moyenne =", inter_per_user.mean())

    return R, user_to_idx, item_to_idx, idx_to_item


# =========================================================
# 4. SOUPE TEXTE
# =========================================================

def make_soup(items_df: pd.DataFrame, item_to_idx: dict):
    # Garder uniquement les items présents dans les interactions
    df = items_df[items_df["i"].isin(item_to_idx.keys())].copy()

    # Aligner sur l'ordre des colonnes de R
    df["col_idx"] = df["i"].map(item_to_idx)
    df = df.sort_values("col_idx").reset_index(drop=True)

    # Nettoyage
    for col in ["Title", "Author", "Publisher", "Subjects", "summary"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""

    # Construire la soupe
    df["soup"] = (
        df["Title"].astype(str) + " " +
        df["Author"].astype(str) + " " +
        df["Publisher"].astype(str) + " " +
        (df["Subjects"].astype(str) + " ") * 3 +
        (df["summary"].astype(str) + " ") * 2
    )

    print("📐 Soupe shape :", df.shape)
    print("📝 Exemple de soupe :", df["soup"].iloc[0][:200], "...")
    return df


# =========================================================
# 5. SIMILARITÉS
# =========================================================

def build_content_sim(df_soup: pd.DataFrame):
    tfidf = TfidfVectorizer(
        max_features=20000,
        min_df=3,
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(df_soup["soup"])
    print("🔤 TF-IDF matrix:", tfidf_matrix.shape)

    sim = cosine_similarity(tfidf_matrix, dense_output=True)
    np.fill_diagonal(sim, 0.0)
    print("✅ Sim contenu shape:", sim.shape)
    return sim


def build_cf_sim(R: csr_matrix):
    print("🧮 Similarité CF item-item (cosine)...")
    sim = cosine_similarity(R.T, dense_output=True)
    np.fill_diagonal(sim, 0.0)
    print("✅ Sim CF shape:", sim.shape)
    return sim


# =========================================================
# 6. MAP@10
# =========================================================

def map10(test: pd.DataFrame,
          R: csr_matrix,
          sim_cf: np.ndarray,
          sim_content: np.ndarray,
          user_to_idx: dict,
          idx_to_item: dict,
          alpha: float) -> float:
    """
    MAP@10 hybride : alpha * CF + (1-alpha) * Contenu
    """
    print(f"\n🐱 Évaluation MAP@10 — alpha={alpha:.2f} (CF), {1-alpha:.2f} (contenu)")

    # Ground truth
    gt = test.groupby("u")["i"].apply(set).to_dict()
    users = list(gt.keys())
    aps = []

    for u in tqdm(users, desc="Users"):
        if u not in user_to_idx:
            aps.append(0.0)
            continue

        u_idx = user_to_idx[u]
        row = R.getrow(u_idx)

        if row.nnz == 0:
            aps.append(0.0)
            continue

        # ⚠️ Conversion explicite en ndarray
        sc_cf = np.asarray(row.dot(sim_cf)).ravel()
        sc_cb = np.asarray(row.dot(sim_content)).ravel()

        scores = alpha * sc_cf + (1.0 - alpha) * sc_cb

        # Masquer les items déjà vus
        seen = row.indices
        scores[seen] = -np.inf

        # Top-K
        top_idx = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        rec_items = [idx_to_item[i] for i in top_idx]

        true_items = gt[u]
        hits = 0
        sum_prec = 0.0

        for rank, it in enumerate(rec_items, start=1):
            if it in true_items:
                hits += 1
                sum_prec += hits / rank

        if len(true_items) == 0:
            aps.append(0.0)
        else:
            aps.append(sum_prec / min(len(true_items), TOP_K))

    return float(np.mean(aps))


# =========================================================
# 7. MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    items_df, interactions, sample_sub = load_data()

    # --- Split LOO + gestion multi-interactions ---
    train_df, test_df = leave_one_out_split(interactions)

    # --- Matrice R ---
    R, user_to_idx, item_to_idx, idx_to_item = build_R(train_df)

    # --- Contenu ---
    df_soup = make_soup(items_df, item_to_idx)
    sim_content = build_content_sim(df_soup)

    # --- CF ---
    sim_cf = build_cf_sim(R)

    if args.evaluate:
        best_score = -1
        best_alpha = None

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            score = map10(test_df, R, sim_cf, sim_content, user_to_idx, idx_to_item, alpha)
            print(f"✅ MAP@10 (alpha={alpha}) = {score:.5f}")
            if score > best_score:
                best_score = score
                best_alpha = alpha

        print("\n" + "="*60)
        print(f"🔥 Meilleur MAP@10 (LOO) = {best_score:.5f} avec alpha = {best_alpha}")
        print("="*60)
    else:
        print("Ajoute --evaluate pour lancer l'évaluation.")


if __name__ == "__main__":
    main()