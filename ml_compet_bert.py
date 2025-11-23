# ml_compet_bert.py
# Reco hybride CF + SBERT avec ton fichier enrichi books_enriched_BASE.xlsx

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG FICHIERS
# =========================
ITEMS_FILE = "books_enriched_BASE.xlsx"   # ton XLSX enrichi
ITEMS_SHEET = "Sheet1"                    # adapte si le nom de l'onglet change

INTERACTIONS_FILE = "kaggle_data/interactions_train.csv"
SAMPLE_SUB_FILE = "kaggle_data/sample_submission.csv"

TOP_K = 10


# =========================
# 1. CHARGEMENT DONNÉES
# =========================

def load_data():
    print(f"📘 Chargement items depuis {ITEMS_FILE}")
    items_df = pd.read_excel(ITEMS_FILE, sheet_name=ITEMS_SHEET)

    print(f"📘 Chargement interactions depuis {INTERACTIONS_FILE}")
    interactions = pd.read_csv(INTERACTIONS_FILE)

    print(f"📘 Chargement sample_submission depuis {SAMPLE_SUB_FILE}")
    sample_sub = pd.read_csv(SAMPLE_SUB_FILE)

    print(f"Items : {items_df.shape}")
    print(f"Interactions : {interactions.shape}")
    print(f"Sample sub : {sample_sub.shape}")

    return items_df, interactions, sample_sub


# =========================
# 2. SPLIT TEMPOREL TRAIN / TEST
# =========================

def train_test_split_by_time(interactions, frac=0.8):
    """
    Split 80/20 par utilisateur basé sur le temps t.
    """
    df = interactions.sort_values(["u", "t"]).copy()
    df["pct_rank"] = df.groupby("u")["t"].rank(pct=True, method="dense")

    train = df[df["pct_rank"] <= frac].copy()
    test = df[df["pct_rank"] > frac].copy()

    train.drop(columns=["pct_rank"], inplace=True)
    test.drop(columns=["pct_rank"], inplace=True)

    print(f"Train: {train.shape}, Test: {test.shape}")

    train_users = set(train["u"].unique())
    test_users = set(test["u"].unique())
    ghost_users = test_users - train_users
    print(f"👻 Users dans le test mais pas dans le train : {len(ghost_users)}")

    return train, test


# =========================
# 3. MATRICE UTILISATEUR-ITEM
# =========================

def build_interaction_matrix(train_df):
    """
    Construit R (users x items) en CSR + mappings.
    """
    users = sorted(train_df["u"].unique())
    items = sorted(train_df["i"].unique())

    user_to_idx = {u: idx for idx, u in enumerate(users)}
    item_to_idx = {i: idx for idx, i in enumerate(items)}
    idx_to_user = {idx: u for u, idx in user_to_idx.items()}
    idx_to_item = {idx: i for i, idx in item_to_idx.items()}

    row_idx = train_df["u"].map(user_to_idx).values
    col_idx = train_df["i"].map(item_to_idx).values
    data = np.ones(len(train_df), dtype=np.float32)

    R = csr_matrix((data, (row_idx, col_idx)), shape=(len(users), len(items)))

    print(f"📊 R : {R.shape[0]} users x {R.shape[1]} items")
    print(f"✅ Matrice R construite, nnz = {R.nnz}")

    return R, user_to_idx, item_to_idx, idx_to_user, idx_to_item


# =========================
# 4. SOUPE DE TEXTE (CONTENU)
# =========================

def make_soup_df(items_df, item_to_idx):
    """
    Aligne items_df sur les items présents dans les interactions
    et construit une colonne 'soup' qui concatène toutes les features.
    """
    # On garde uniquement les items présents dans item_to_idx (colonnes de R)
    items_df = items_df[items_df["i"].isin(item_to_idx.keys())].copy()

    # Assure le même ordre que les colonnes de R
    items_df["col_idx"] = items_df["i"].map(item_to_idx)
    items_df = items_df.sort_values("col_idx").reset_index(drop=True)

    # Remplissage des colonnes texte
    for col in ["Title", "Author", "Publisher", "Subjects", "summary", "language"]:
        if col in items_df.columns:
            items_df[col] = items_df[col].fillna("")
        else:
            items_df[col] = ""

    # ----- Tags pages -----
    def page_bucket(p):
        try:
            p = float(p)
        except (TypeError, ValueError):
            return "pages_unknown"
        if p == 0:
            return "pages_unknown"
        if p <= 150:
            return "pages_short"
        if p <= 300:
            return "pages_medium"
        if p <= 600:
            return "pages_long"
        return "pages_huge"

    if "page_count" in items_df.columns:
        items_df["page_tag"] = items_df["page_count"].apply(page_bucket)
    else:
        items_df["page_tag"] = "pages_unknown"

    # ----- Tags année -----
    if "published_year" in items_df.columns:
        def year_tag(y):
            if pd.isna(y):
                return "year_unknown"
            s = str(y)[:4]
            if not s.isdigit():
                return "year_unknown"
            return f"year_{s}"
        items_df["year_tag"] = items_df["published_year"].apply(year_tag)
    else:
        items_df["year_tag"] = "year_unknown"

    # ----- Tags langue -----
    def lang_tag(l):
        if not isinstance(l, str) or not l:
            return "lang_unknown"
        l = l.lower()
        if l.startswith("/languages/"):
            l = l.split("/")[-1]
        return f"lang_{l}"

    items_df["lang_tag"] = items_df["language"].apply(lang_tag)

    # ----- Soupe globale -----
    def build_soup(row):
        title = row["Title"]
        author = row["Author"]
        publisher = row["Publisher"]
        subjects = (row["Subjects"] + " ") * 3
        summary = (row["summary"] + " ") * 2
        lang = row["lang_tag"]
        year = row["year_tag"]
        pages = row["page_tag"]
        return f"{title} {author} {publisher} {subjects}{summary}{lang} {year} {pages}"

    items_df["soup"] = items_df.apply(build_soup, axis=1)

    print(f"📐 items_df aligné : {items_df.shape}")
    print("🍲 Exemple de soupe :", items_df["soup"].iloc[0][:200], "...")
    return items_df


def build_content_similarity_sbert(items_df):
    """
    Encode la soupe avec SBERT et calcule la similarité cosine item-item.
    """
    texts = items_df["soup"].tolist()

    print("🧠 Encodage SBERT des items (paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    print(f"Embeddings shape : {embeddings.shape}")

    print("🤝 Similarité contenu (SBERT, cosine)...")
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 0.0)
    print(f"✅ Matrice contenu SBERT prête : {sim.shape}")
    return sim


# =========================
# 5. CF ITEM-ITEM
# =========================

def build_cf_similarity(R):
    """
    CF item-item : similarité cosine dense sur R.T
    """
    print("🧮 Similarité CF item-item (cosine dense)...")
    sim = cosine_similarity(R.T, dense_output=True)
    np.fill_diagonal(sim, 0.0)
    print(f"✅ Matrice CF prête : {sim.shape}")
    return sim


# =========================
# 6. ÉVALUATION MAP@10
# =========================

def map_at_10(test_df, R_train, sim_cf, sim_content, user_to_idx, idx_to_item, alpha):
    """
    alpha = poids du CF, (1 - alpha) = poids du contenu SBERT.
    MAP@10 sur l'ensemble du test.
    """
    print(f"\n🐱 Évaluation hybride (alpha CF = {alpha:.2f}, SBERT = {1-alpha:.2f})")

    # ground truth : {user: set(items dans le test)}
    gt = test_df.groupby("u")["i"].apply(set).to_dict()
    users_eval = [u for u in gt.keys() if u in user_to_idx]

    aps = []

    for u in tqdm(users_eval, desc="Users", leave=False):
        u_idx = user_to_idx[u]
        user_row = R_train.getrow(u_idx)

        if user_row.nnz == 0:
            # user sans historique dans le train
            aps.append(0.0)
            continue

        # scores CF et contenu
        scores_cf = np.asarray(user_row.dot(sim_cf)).ravel()
        scores_content = np.asarray(user_row.dot(sim_content)).ravel()

        scores = alpha * scores_cf + (1 - alpha) * scores_content

        # On enlève les items déjà vus
        seen = user_row.indices
        scores[seen] = -np.inf

        # Top K
        if np.all(~np.isfinite(scores)):
            aps.append(0.0)
            continue

        top_idx = np.argpartition(scores, -TOP_K)[-TOP_K:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        rec_items = [idx_to_item[i] for i in top_idx]
        true_items = gt[u]

        hits = 0
        sum_prec = 0.0
        for rank, item in enumerate(rec_items, start=1):
            if item in true_items:
                hits += 1
                sum_prec += hits / rank

        if len(true_items) == 0:
            ap = 0.0
        else:
            ap = sum_prec / min(len(true_items), TOP_K)
        aps.append(ap)

    return float(np.mean(aps))


# =========================
# 7. GÉNÉRATION SUBMISSION
# =========================

def generate_submission(R, sim_cf, sim_content, alpha, sample_sub, user_to_idx, idx_to_item):
    """
    Produit submission.csv avec recommandations pour chaque user_id
    (10 items séparés par des espaces).
    """
    print(f"\n📦 Génération de la submission finale avec alpha = {alpha}")
    user_ids_sub = sample_sub["user_id"].values

    # popularité pour fallback users sans historique
    item_popularity = np.array(R.sum(axis=0)).flatten()
    popular_items_idx = item_popularity.argsort()[::-1]

    recs_for_users = []

    for u in tqdm(user_ids_sub, desc="Submission users"):
        if u in user_to_idx:
            u_idx = user_to_idx[u]
            user_row = R.getrow(u_idx)

            if user_row.nnz > 0:
                scores_cf = np.asarray(user_row.dot(sim_cf)).ravel()
                scores_content = np.asarray(user_row.dot(sim_content)).ravel()
                scores = alpha * scores_cf + (1 - alpha) * scores_content

                seen = user_row.indices
                scores[seen] = -np.inf

                if np.all(~np.isfinite(scores)):
                    rec_idx = popular_items_idx[:TOP_K].tolist()
                else:
                    top_idx = np.argpartition(scores, -TOP_K)[-TOP_K:]
                    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
                    rec_idx = top_idx.tolist()
            else:
                # user sans historique → items les plus populaires
                rec_idx = popular_items_idx[:TOP_K].tolist()
        else:
            # user inconnu (en théorie n'existe pas dans sample_submission)
            rec_idx = popular_items_idx[:TOP_K].tolist()

        rec_item_ids = [str(idx_to_item[i]) for i in rec_idx]
        recs_for_users.append(" ".join(rec_item_ids))

    out = sample_sub.copy()
    out["recommendation"] = recs_for_users
    out.to_csv("submission.csv", index=False)
    print("✅ Fichier 'submission.csv' généré !")


# =========================
# 8. MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Faire un split train/test temporel et calculer MAP@10"
    )
    parser.add_argument(
        "--submission",
        action="store_true",
        help="Entraîner sur toutes les données et générer submission.csv"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Poids CF pour le mode --submission (0 = full SBERT, 1 = full CF)"
    )
    args = parser.parse_args()

    items_df, interactions, sample_sub = load_data()

    if args.evaluate:
        # --- 1) Split temporel ---
        train_df, test_df = train_test_split_by_time(interactions, frac=0.8)

        # --- 2) Matrice R sur le TRAIN uniquement ---
        R_train, user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_interaction_matrix(train_df)

        # --- 3) Contenu (SBERT) aligné sur R_train ---
        items_for_sim = make_soup_df(items_df, item_to_idx)
        sim_content = build_content_similarity_sbert(items_for_sim)

        # --- 4) CF item-item ---
        sim_cf = build_cf_similarity(R_train)

        # --- 5) Grid search sur alpha ---
        best_alpha = None
        best_map = -1.0
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            score = map_at_10(test_df, R_train, sim_cf, sim_content, user_to_idx, idx_to_item, alpha)
            print(f"✅ MAP@10 (alpha={alpha}) = {score:.5f}")
            if score > best_map:
                best_map = score
                best_alpha = alpha

        print("\n" + "=" * 60)
        print(f"🔥 Meilleur MAP@10 (SBERT) = {best_map:.5f} avec alpha = {best_alpha}")
        print("=" * 60)

        # --- 6) Ré-entraînement full + submission ---
        print("\n📦 Ré-entraînement sur toutes les interactions pour submission...")
        R_full, user_to_idx_f, item_to_idx_f, _, idx_to_item_f = build_interaction_matrix(interactions)
        items_full = make_soup_df(items_df, item_to_idx_f)
        sim_content_full = build_content_similarity_sbert(items_full)
        sim_cf_full = build_cf_similarity(R_full)

        generate_submission(
            R_full, sim_cf_full, sim_content_full, best_alpha,
            sample_sub, user_to_idx_f, idx_to_item_f
        )

    elif args.submission:
        # --- Mode direct submission sur toutes les interactions ---
        R_full, user_to_idx, item_to_idx, _, idx_to_item = build_interaction_matrix(interactions)
        items_full = make_soup_df(items_df, item_to_idx)
        sim_content_full = build_content_similarity_sbert(items_full)
        sim_cf_full = build_cf_similarity(R_full)

        generate_submission(
            R_full, sim_cf_full, sim_content_full, args.alpha,
            sample_sub, user_to_idx, idx_to_item
        )
    else:
        print("Utilisation :")
        print("  python3 ml_compet_bert.py --evaluate")
        print("  python3 ml_compet_bert.py --submission --alpha 0.5")


if __name__ == "__main__":
    main()