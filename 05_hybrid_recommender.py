import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD  # Importation pour l'optimisation TF-IDF (LSI)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import sys

# --- HYPERPARAMÈTRES GLOBALES ---
SUBJECT_WEIGHT_FACTOR = 3  # Poids des sujets pour SBERT (doit être >= 1)
TFIDF_SUBJECT_WEIGHT = 5  # Poids des sujets pour TF-IDF
TFIDF_SVD_COMPONENTS = 500  # Dimensions latentes pour le TF-IDF (LSI)
EMBEDDINGS_FILE = 'sbert_embeddings_ultimate.npy'

# ==============================================================================
# SECTION 0: CHARGEMENT ET SPLIT (Vérifié et Corrigé)
# ==============================================================================

print("\n--- 0. CHARGEMENT ET SPLIT ---")

# 1. Chargement des données brutes
try:
    # Utilisation des chemins relatifs corrigés
    interactions_df = pd.read_csv('kaggle_data/interactions_train.csv')
    items_df = pd.read_csv('kaggle_data/items.csv')
except FileNotFoundError:
    print("ERREUR FATALE: Le chemin 'kaggle_data/' est incorrect. Vérifiez le chemin ou la présence des fichiers.")
    sys.exit()

# 2. Renommage et Nettoyage
if 'user_id' in interactions_df.columns:
    interactions_df = interactions_df.rename(columns={'user_id': 'u', 'item_id': 'i', 'timestamp': 't'})

# 3. Le Split Temporel (Méthode 80/20 par timestamp)
interactions_df = interactions_df.sort_values(["u", "t"])
interactions_df["pct_rank"] = interactions_df.groupby("u")["t"].rank(pct=True, method='dense')

train_data = interactions_df[interactions_df["pct_rank"] < 0.8]
test_data = interactions_df[interactions_df["pct_rank"] >= 0.8]

print("✅ Données chargées et splittées correctement.")

# ==============================================================================
# SECTION 1: MATRICE FONDATION (R_train)
# ==============================================================================

print("\n--- 1. CRÉATION DE R_train (User x Item) ---")

# MAPPING (Basé sur l'univers complet des IDs pour la cohérence)
unique_users = interactions_df['u'].unique()
unique_items = interactions_df['i'].unique()
user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
idx_to_item = {idx: i for i, idx in item_to_idx.items()}

train_rows = train_data['u'].map(user_to_idx).values
train_cols = train_data['i'].map(item_to_idx).values
interactions = np.ones(len(train_data))

R_train = csr_matrix((interactions, (train_rows, train_cols)),
                     shape=(len(unique_users), len(unique_items)))

print(f"✅ R_train (CF Base) construite. Shape: {R_train.shape}")

# ==============================================================================
# SECTION 2: MODÈLE A - ITEM-ITEM CF (Comportement)
# ==============================================================================

print("\n--- 2. MODÈLE A: Similarité CF (Interactions) ---")
item_sim_matrix_cf = cosine_similarity(R_train.T, dense_output=True)
np.fill_diagonal(item_sim_matrix_cf, 0)
print(f"✅ Matrice A (CF) prête. Shape: {item_sim_matrix_cf.shape}")

# ==============================================================================
# SECTION 3: PRÉPARATION DU CONTENU (Optimisations)
# ==============================================================================

print("\n--- 3. PRÉPARATION DU CONTENU (Soupes de mots optimisées) ---")

# 1. Nettoyage des colonnes SÛRES
items_df['Title'] = items_df['Title'].fillna('')
items_df['Author'] = items_df['Author'].fillna('')
items_df['Subjects'] = items_df['Subjects'].fillna('')


# NOTE: Si 'summary' existe, la charger ici: items_df['summary'] = items_df['summary'].fillna('')

# 2. Création de la SOUPE TF-IDF (Optimisation : Poids et Séparateur)
def create_tfidf_soup(x):
    # Répéter les sujets 5 fois (le poids maximal)
    return (x['Title'] + ' ') + (x['Author'] + ' ') + (x['Subjects'] + ' ') * TFIDF_SUBJECT_WEIGHT


items_df['tfidf_soup'] = items_df.apply(create_tfidf_soup, axis=1)


# 3. Création de la SOUPE SBERT (Optimisation : Format lisible par BERT)
def create_sbert_soup(x):
    # Répéter les sujets 3 fois (pondération)
    subjects_text = (x['Subjects'] + ' ') * SUBJECT_WEIGHT_FACTOR
    # Utilisation d'un format clair pour le modèle sémantique
    # NOTE: Adapter si vous ajoutez 'summary' (ex: f"...[SUMMARY] {x['summary']}")
    return f"{x['Title']} [SEP] {x['Author']} [SEP] {subjects_text}"


items_df['sbert_soup'] = items_df.apply(create_sbert_soup, axis=1)

print("✅ Soupes TF-IDF et SBERT créées.")

# ==============================================================================
# SECTION 4: MODÈLES B & C - MATRICES DE SIMILARITÉ
# ==============================================================================

# --- MODÈLE B: TF-IDF + SVD (Latent Semantic Indexing - NOUVEAU CHAMPION) ---
print("\n--- 4A. MODÈLE B: Matrice TF-IDF + SVD (Optimisé) ---")


# On utilise un custom tokenizer pour les sujets (séparés par ;) et on ignore les mots trop rares (min_df=5)
def custom_tokenizer(text):
    # Sépare par espace ou point-virgule, puis enlève le vide
    return [t.strip() for t in text.replace(';', ' ').split() if t.strip()]


tfidf = TfidfVectorizer(tokenizer=custom_tokenizer, min_df=5, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(items_df['tfidf_soup'])

# Application de la Réduction de Dimension (LSI)
svd_transformer = TruncatedSVD(n_components=TFIDF_SVD_COMPONENTS, random_state=42)
tfidf_svd_matrix = svd_transformer.fit_transform(tfidf_matrix)

# Calcul de la Similarité sur les vecteurs SVD (plus court, moins bruité)
item_sim_matrix_tfidf_svd = cosine_similarity(tfidf_svd_matrix, tfidf_svd_matrix)
np.fill_diagonal(item_sim_matrix_tfidf_svd, 0)
print(f"✅ Matrice B (TF-IDF+SVD) prête. Shape: {item_sim_matrix_tfidf_svd.shape}")

# --- MODÈLE C: SBERT (Sémantique) ---
print("\n--- 4B. MODÈLE C: Matrice SBERT (Nécessite GPU/Caching) ---")

if os.path.exists(EMBEDDINGS_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"✅ Embeddings SBERT chargés ({embeddings.shape}).")
else:
    print("⚠️ Encoudage SBERT lancé (vérifiez l'activation du GPU).")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(items_df['sbert_soup'].tolist(), show_progress_bar=True, convert_to_tensor=False)
    np.save(EMBEDDINGS_FILE, embeddings)
    print("✅ Embeddings sauvegardés.")

item_sim_matrix_sbert = cosine_similarity(embeddings)
np.fill_diagonal(item_sim_matrix_sbert, 0)
print(f"✅ Matrice C (SBERT) prête. Shape: {item_sim_matrix_sbert.shape}")

# ==============================================================================
# SECTION 5: ALIGNEMENT DES MATRICES B & C SUR LA MATRICE A
# ==============================================================================

print("\n--- 5. ALIGNEMENT DES MATRICES ---")
# 1. Créer le plan d'alignement
content_id_to_row_idx = pd.Series(items_df.index, index=items_df['i']).to_dict()
n_items_cf = len(idx_to_item)
aligned_content_indices = []

for cf_idx in range(n_items_cf):
    item_id = idx_to_item[cf_idx]
    # Robustesse : Vérification de l'existence de l'ID dans le contenu
    if item_id in content_id_to_row_idx:
        content_idx = content_id_to_row_idx[item_id]
        aligned_content_indices.append(content_idx)
    else:
        # Si un livre est dans le Train CF mais pas dans items_df, c'est un problème.
        # On suppose qu'ils sont tous là pour l'instant.
        pass

    # 2. Appliquer l'alignement sur B (TF-IDF+SVD) et C (SBERT)
# NOTE: Utilisation de la nouvelle matrice optimisée!
content_sim_aligned_tfidf_svd = item_sim_matrix_tfidf_svd[np.ix_(aligned_content_indices, aligned_content_indices)]
content_sim_aligned_sbert = item_sim_matrix_sbert[np.ix_(aligned_content_indices, aligned_content_indices)]

print("✅ Matrices B (TFIDF+SVD) et C (SBERT) alignées et prêtes pour la fusion.")

# ==============================================================================
# SECTION 6 & 7: ÉVALUATION ET TUNING FINAL (Logique inchangée, RAM-SAFE)
# ==============================================================================

# On garde les fonctions RAM-SAFE.
# (Code des fonctions get_user_recommendations_safe et evaluate_map_at_10_3_MODELS)

item_popularity = np.array(R_train.sum(axis=0)).flatten()
popular_indices = item_popularity.argsort()[-10:][::-1]


def get_user_recommendations_safe(user_idx, interaction_matrix, sim_matrix, top_k=10):
    user_history = interaction_matrix.getrow(user_idx)
    recommendations = []

    if user_history.nnz > 0:
        scores = user_history.dot(sim_matrix)

        if hasattr(scores, 'toarray'):
            scores = scores.toarray().flatten()
        else:
            scores = scores.flatten()

        seen_indices = user_history.indices
        scores[seen_indices] = -np.inf

        top_indices_hybride = scores.argsort()[-top_k:][::-1]

        for idx in top_indices_hybride:
            if scores[idx] > 0: recommendations.append(idx)

    for pop_idx in popular_indices:
        if len(recommendations) >= top_k: break
        if (pop_idx not in recommendations) and (pop_idx not in user_history.indices):
            recommendations.append(pop_idx)

    return recommendations[:top_k]


def evaluate_map_at_10_3_MODELS(test_df, R_train,
                                matrix_A_cf, matrix_B_tfidf_svd, matrix_C_sbert,
                                user_to_idx, idx_to_item,
                                weights, top_k=10):
    alpha, beta, gamma = weights

    test_ground_truth = test_df.groupby('u')['i'].apply(set).to_dict()
    average_precisions = []
    common_users = [u for u in test_ground_truth.keys() if u in user_to_idx]

    for user_id in tqdm(common_users, desc=f"Eval w={weights}", leave=False):
        u_idx = user_to_idx[user_id]
        user_history = R_train.getrow(u_idx)
        recommendations = []

        if user_history.nnz > 0:
            # Calcul des scores individuels
            scores_cf = user_history.dot(matrix_A_cf).flatten()
            scores_tfidf = user_history.dot(matrix_B_tfidf_svd).flatten()  # <--- Nouvelle matrice optimisée
            scores_sbert = user_history.dot(matrix_C_sbert).flatten()

            # Fusion
            scores_hybrid = (alpha * scores_cf) + (beta * scores_tfidf) + (gamma * scores_sbert)

            # Exclusion et Top-K
            seen_indices = user_history.indices
            scores_hybrid[seen_indices] = -np.inf
            top_indices_hybride = scores_hybrid.argsort()[-top_k:][::-1]
            for idx in top_indices_hybride:
                if scores_hybrid[idx] > 0: recommendations.append(idx)

        # Roue de secours (Popularité)
        for pop_idx in popular_indices:
            if len(recommendations) >= top_k: break
            if (pop_idx not in recommendations) and (pop_idx not in user_history.indices):
                recommendations.append(pop_idx)

        rec_items = [idx_to_item[i] for i in recommendations[:top_k]]
        true_items = test_ground_truth[user_id]
        hits = 0
        sum_precisions = 0
        for rank, item in enumerate(rec_items, start=1):
            if item in true_items:
                hits += 1
                sum_precisions += hits / rank

        if not true_items:
            ap = 0
        else:
            ap = sum_precisions / min(len(true_items), 10)
        average_precisions.append(ap)

    return np.mean(average_precisions)


# --- SECTION 7: TUNING FINAL ---

combinations_to_test = [
    [0.1, 0.8, 0.1],  # Priorité TFIDF+SVD (Ancien champion)
]

best_score = 0
best_weights = []

print("\n\n--- 🚀 LANCEMENT DU GRAND TEST DE FUSION (A+B+C) ---")

for weights in combinations_to_test:
    score_hybrid = evaluate_map_at_10_3_MODELS(
        test_data, R_train,
        item_sim_matrix_cf, content_sim_aligned_tfidf_svd, content_sim_aligned_sbert,
        # <--- Utilisation des matrices optimisées
        user_to_idx, idx_to_item,
        weights
    )

    print(f"🏆 Score obtenu (w={weights}): {score_hybrid:.5f}")

    if score_hybrid > best_score:
        best_score = score_hybrid
        best_weights = weights

print("\n" + "=" * 70)
print(f"🔥🔥🔥 MEILLEUR SCORE HYBRIDE : {best_score:.5f}")
print(f"       (Obtenu avec [CF, TF-IDF+SVD, SBERT] = {best_weights})")
print(f"       Baseline à battre : 0.1452")
print("=" * 70)