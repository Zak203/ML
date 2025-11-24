# ml_compet_submit.py

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

# =====================================================
# 1. LOAD DATA
# =====================================================

def load_data():
    print(f"📘 Items : {ITEMS_FILE}")
    items = pd.read_excel(ITEMS_FILE, sheet_name=ITEMS_SHEET)

    print(f"📘 Interactions : {INTERACTIONS_FILE}")
    inter = pd.read_csv(INTERACTIONS_FILE)

    print(f"📘 Sample Sub : {SAMPLE_SUB_FILE}")
    sample = pd.read_csv(SAMPLE_SUB_FILE)

    print("Items :", items.shape)
    print("Interactions :", inter.shape)
    return items, inter, sample


# =====================================================
# 2. LEAVE ONE OUT
# =====================================================

def leave_one_out(interactions):
    df = interactions.sort_values(["u", "t"]).copy()

    before = df.shape[0]
    df = df.drop_duplicates(subset=["u", "i"])
    after = df.shape[0]
    print(f"🧹 Drop doublons (u,i) : {before} -> {after}")

    test = df.groupby("u").tail(1)
    train = df.drop(test.index)

    print("Train:", train.shape, "Test:", test.shape)
    return train, test


# =====================================================
# 3. MATRICE R
# =====================================================

def build_R(train_df):
    users = sorted(train_df["u"].unique())
    items = sorted(train_df["i"].unique())

    user_to_idx = {u: idx for idx, u in enumerate(users)}
    item_to_idx = {it: idx for idx, it in enumerate(items)}
    idx_to_item = {idx: it for it, idx in item_to_idx.items()}

    rows = train_df["u"].map(user_to_idx).values
    cols = train_df["i"].map(item_to_idx).values
    data = np.ones(len(train_df), dtype=np.float32)

    R = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    print(f"📊 R = {R.shape}, nnz={R.nnz}")

    return R, user_to_idx, item_to_idx, idx_to_item


# =====================================================
# 4. SOUPE
# =====================================================

def make_soup(items_df, item_to_idx):
    df = items_df[items_df["i"].isin(item_to_idx.keys())].copy()
    df["col_idx"] = df["i"].map(item_to_idx)
    df = df.sort_values("col_idx").reset_index(drop=True)

    for col in ["Title", "Author", "Publisher", "Subjects", "summary"]:
        df[col] = df[col].fillna("")

    df["soup"] = (
        df["Title"].astype(str) + " " +
        df["Author"].astype(str) + " " +
        df["Publisher"].astype(str) + " " +
        (df["Subjects"].astype(str) + " ") * 3 +
        (df["summary"].astype(str) + " ") * 2
    )

    print("📐 Soupe shape :", df.shape)
    return df


# =====================================================
# 5. SIMILARITES
# =====================================================

def build_sim_content(df_soup):
    tfidf = TfidfVectorizer(max_features=20000, min_df=3, ngram_range=(1,2))
    X = tfidf.fit_transform(df_soup["soup"])
    print("🔤 TF-IDF :", X.shape)

    sim = cosine_similarity(X, dense_output=True)
    np.fill_diagonal(sim, 0)
    print("✅ Sim contenu :", sim.shape)
    return sim


def build_sim_cf(R):
    sim = cosine_similarity(R.T, dense_output=True)
    np.fill_diagonal(sim, 0)
    print("🧮 Sim CF :", sim.shape)
    return sim


# =====================================================
# 6. MAP@10
# =====================================================

def map10(test, R, sim_cf, sim_content, user_to_idx, idx_to_item, alpha):
    gt = test.groupby("u")["i"].apply(set).to_dict()
    users = list(gt.keys())
    aps = []

    for u in tqdm(users):
        if u not in user_to_idx:
            aps.append(0)
            continue

        u_idx = user_to_idx[u]
        row = R.getrow(u_idx)

        if row.nnz == 0:
            aps.append(0)
            continue

        sc_cf = np.asarray(row.dot(sim_cf)).ravel()
        sc_ct = np.asarray(row.dot(sim_content)).ravel()

        scores = alpha * sc_cf + (1 - alpha) * sc_ct
        scores[row.indices] = -np.inf

        top = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top = top[np.argsort(scores[top])[::-1]]
        rec = [idx_to_item[i] for i in top]

        hits = 0
        s = 0
        for r, it in enumerate(rec, start=1):
            if it in gt[u]:
                hits += 1
                s += hits / r

        aps.append(s / min(len(gt[u]), TOP_K))

    return float(np.mean(aps))


# =====================================================
# 7. SUBMISSION
# =====================================================

def generate_submission(R, sim_cf, sim_content, alpha, sample_sub, user_to_idx, idx_to_item):
    print("📦 Génération submission...")

    item_pop = np.array(R.sum(axis=0)).ravel()
    popular_idx = item_pop.argsort()[::-1]

    recs = []

    for u in tqdm(sample_sub["user_id"]):
        if u in user_to_idx:
            u_idx = user_to_idx[u]
            row = R.getrow(u_idx)

            if row.nnz > 0:
                sc_cf = np.asarray(row.dot(sim_cf)).ravel()
                sc_ct = np.asarray(row.dot(sim_content)).ravel()
                scores = alpha * sc_cf + (1 - alpha) * sc_ct
                scores[row.indices] = -np.inf

                top = np.argpartition(scores, -TOP_K)[-TOP_K:]
                top = top[np.argsort(scores[top])[::-1]]
            else:
                top = popular_idx[:TOP_K]
        else:
            top = popular_idx[:TOP_K]

        recs.append(" ".join(str(idx_to_item[i]) for i in top))

    out = sample_sub.copy()
    out["recommendation"] = recs
    out.to_csv("submission.csv", index=False)
    print("✅ submission.csv généré !")


# =====================================================
# 8. MAIN
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    items, interactions, sample = load_data()
    train, test = leave_one_out(interactions)

    R, user_to_idx, item_to_idx, idx_to_item = build_R(train)
    df_soup = make_soup(items, item_to_idx)
    sim_content = build_sim_content(df_soup)
    sim_cf = build_sim_cf(R)

    if args.evaluate:
        for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
            score = map10(test, R, sim_cf, sim_content, user_to_idx, idx_to_item, a)
            print(f"MAP@10 (alpha={a}) = {score:.5f}")

    if args.submission:
        generate_submission(R, sim_cf, sim_content, args.alpha,
                            sample, user_to_idx, idx_to_item)


if __name__ == "__main__":
    main()