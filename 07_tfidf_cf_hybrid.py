import pandas as pd
import numpy as np
import sys
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INTERACTIONS_PATH = 'kaggle_data/interactions_train.csv'
ITEMS_PATH = 'kaggle_data/items.csv'
BASELINE_SCORE = 0.1452

# Hyperparamètres optimisés
N_LATENT_FACTORS = 150  # Augmenté pour capturer plus de patterns
ALS_REGULARIZATION = 0.01
ALS_ITERATIONS = 15
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.8

print("=" * 80)
print("🚀 SYSTÈME DE RECOMMANDATION HYBRIDE OPTIMISÉ")
print("=" * 80)

# ==============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================

print("\n📂 Chargement des données...")
try:
    interactions_df = pd.read_csv(INTERACTIONS_PATH)
    items_df = pd.read_csv(ITEMS_PATH)
except FileNotFoundError as e:
    print(f"❌ ERREUR: {e}")
    sys.exit(1)

# Renommage
if 'user_id' in interactions_df.columns:
    interactions_df = interactions_df.rename(columns={
        'user_id': 'u',
        'item_id': 'i',
        'timestamp': 't'
    })

print(f"   Users: {interactions_df['u'].nunique():,}")
print(f"   Items: {interactions_df['i'].nunique():,}")
print(f"   Interactions: {len(interactions_df):,}")

# Statistiques de sparsité
n_users = interactions_df['u'].nunique()
n_items = interactions_df['i'].nunique()
sparsity = 100 * (1 - len(interactions_df) / (n_users * n_items))
print(f"   Sparsité: {sparsity:.2f}%")

# Distribution des interactions
user_counts = interactions_df.groupby('u').size()
print(f"   Interactions/user (médiane): {user_counts.median():.0f}")
print(f"   Interactions/user (moyenne): {user_counts.mean():.1f}")

# ==============================================================================
# SPLIT TEMPOREL AMÉLIORÉ
# ==============================================================================

print("\n⏱️  Split temporel (80/20)...")

# Sort global par timestamp
interactions_df = interactions_df.sort_values('t')

# Split par utilisateur pour garder l'ordre temporel
interactions_df["pct_rank"] = interactions_df.groupby("u")["t"].rank(pct=True, method='first')
train_data = interactions_df[interactions_df["pct_rank"] <= 0.8].copy()
test_data = interactions_df[interactions_df["pct_rank"] > 0.8].copy()

print(f"   Train: {len(train_data):,} interactions")
print(f"   Test: {len(test_data):,} interactions")

# Filtrer les utilisateurs avec au moins 1 interaction en train ET test
users_in_train = set(train_data['u'].unique())
users_in_test = set(test_data['u'].unique())
valid_users = users_in_train & users_in_test
print(f"   Utilisateurs valides (train & test): {len(valid_users):,}")

test_data = test_data[test_data['u'].isin(valid_users)]

# ==============================================================================
# CRÉATION DES MAPPINGS ET MATRICE R_train
# ==============================================================================

print("\n🗺️  Création des mappings...")

unique_users = sorted(train_data['u'].unique())
unique_items = sorted(train_data['i'].unique())

user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
idx_to_item = {idx: i for i, idx in item_to_idx.items()}

# Matrice R_train (User x Item) avec poids
train_rows = train_data['u'].map(user_to_idx).values
train_cols = train_data['i'].map(item_to_idx).values

# Poids : plus d'interactions récentes = poids plus élevé
train_data['recency_weight'] = train_data.groupby('u')['t'].rank(pct=True, method='first')
weights = train_data['recency_weight'].values

R_train = csr_matrix(
    (weights, (train_rows, train_cols)),
    shape=(len(unique_users), len(unique_items))
)

print(
    f"   Matrice R_train: {R_train.shape} (sparsité: {100 * (1 - R_train.nnz / (R_train.shape[0] * R_train.shape[1])):.2f}%)")

# Historique utilisateur
user_history_map = train_data.groupby('u')['i'].apply(set).to_dict()

# Popularité (fallback)
item_popularity = np.array(R_train.sum(axis=0)).flatten()
popular_indices = item_popularity.argsort()[-50:][::-1]

# ==============================================================================
# MODÈLE 1: ALS (Alternating Least Squares)
# ==============================================================================

print(f"\n🧮 Entraînement ALS (factors={N_LATENT_FACTORS}, reg={ALS_REGULARIZATION})...")

try:
    from implicit.als import AlternatingLeastSquares

    # ALS nécessite la transposée (Items x Users)
    R_train_T = R_train.T.tocsr()

    als_model = AlternatingLeastSquares(
        factors=N_LATENT_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        random_state=42,
        calculate_training_loss=False
    )

    als_model.fit(R_train_T, show_progress=True)

    # Extraction des facteurs
    user_factors = als_model.user_factors  # (Users, K)
    item_factors = als_model.item_factors  # (Items, K)

    print(f"✅ ALS entraîné: user_factors {user_factors.shape}, item_factors {item_factors.shape}")
    ALS_AVAILABLE = True

except ImportError:
    print("⚠️  Package 'implicit' non disponible. Fallback sur SVD...")
    ALS_AVAILABLE = False

    from sklearn.decomposition import TruncatedSVD

    svd_model = TruncatedSVD(n_components=N_LATENT_FACTORS, random_state=42)
    svd_model.fit(R_train)

    user_factors = svd_model.transform(R_train)  # (Users, K)
    item_factors = svd_model.components_.T  # (Items, K)

    print(f"✅ SVD entraîné: user_factors {user_factors.shape}, item_factors {item_factors.shape}")

# ==============================================================================
# MODÈLE 2: TF-IDF CONTENU (Amélioré)
# ==============================================================================

print("\n📝 Création de la matrice TF-IDF (contenu)...")

# Preprocessing amélioré
items_df['Title'] = items_df['Title'].fillna('').str.lower()
items_df['Author'] = items_df['Author'].fillna('').str.lower()
items_df['Subjects'] = items_df['Subjects'].fillna('').str.lower()


def create_content_soup(row):
    """Combine les features de contenu avec pondération intelligente"""
    title = row['Title']
    author = row['Author']
    subjects = row['Subjects'].replace(';', ' ')

    # Titre 2x, Auteur 1x, Subjects 3x
    return f"{title} {title} {author} {subjects} {subjects} {subjects}"


items_df['content_soup'] = items_df.apply(create_content_soup, axis=1)

# TF-IDF avec meilleurs paramètres
tfidf = TfidfVectorizer(
    min_df=TFIDF_MIN_DF,
    max_df=TFIDF_MAX_DF,
    ngram_range=(1, 2),
    strip_accents='unicode',
    lowercase=True,
    stop_words='english'
)

tfidf_matrix = tfidf.fit_transform(items_df['content_soup'])
print(f"   TF-IDF: {tfidf_matrix.shape} ({len(tfidf.get_feature_names_out())} features)")

# Alignement content -> CF space
content_id_to_row = pd.Series(items_df.index, index=items_df['i']).to_dict()
aligned_content_indices = []

for cf_idx in range(len(idx_to_item)):
    item_id = idx_to_item[cf_idx]
    if item_id in content_id_to_row:
        aligned_content_indices.append(content_id_to_row[item_id])

tfidf_matrix_aligned = tfidf_matrix[aligned_content_indices]
print(f"   TF-IDF aligné: {tfidf_matrix_aligned.shape}")


# ==============================================================================
# FONCTION DE RECOMMANDATION HYBRIDE OPTIMISÉE
# ==============================================================================

def recommend_hybrid_optimized(user_id, alpha_als, alpha_tfidf, top_k=10):
    """
    Génère des recommandations hybrides avec normalisation des scores.

    Args:
        user_id: ID de l'utilisateur
        alpha_als: Poids du modèle ALS/SVD (0-1)
        alpha_tfidf: Poids du modèle TF-IDF (0-1)
        top_k: Nombre de recommandations

    Returns:
        Liste d'item_ids recommandés
    """

    if user_id not in user_to_idx:
        return []

    u_idx = user_to_idx[user_id]
    user_history = user_history_map.get(user_id, set())

    # --- SCORING ALS/SVD ---
    # Prédiction directe via produit matriciel des facteurs latents
    user_vec = user_factors[u_idx]  # (K,)
    scores_als = item_factors.dot(user_vec)  # (Items,)

    # --- SCORING TF-IDF ---
    # Trouver les items lus par l'utilisateur
    user_row = R_train.getrow(u_idx)
    read_indices = user_row.indices

    if len(read_indices) > 0:
        # Profil utilisateur = moyenne des vecteurs TF-IDF des items lus
        user_profile_tfidf = tfidf_matrix_aligned[read_indices].mean(axis=0)
        # Convertir en array numpy si c'est une matrice
        user_profile_tfidf = np.asarray(user_profile_tfidf).flatten()
        # Similarité cosinus avec tous les items
        scores_tfidf_raw = tfidf_matrix_aligned.dot(user_profile_tfidf)
        # Gérer le cas sparse ou dense
        if hasattr(scores_tfidf_raw, 'toarray'):
            scores_tfidf = scores_tfidf_raw.toarray().flatten()
        else:
            scores_tfidf = np.asarray(scores_tfidf_raw).flatten()
    else:
        scores_tfidf = np.zeros(len(item_factors))

    # --- NORMALISATION MIN-MAX ---
    scaler = MinMaxScaler()

    if scores_als.max() > scores_als.min():
        scores_als_norm = scaler.fit_transform(scores_als.reshape(-1, 1)).flatten()
    else:
        scores_als_norm = np.zeros_like(scores_als)

    if scores_tfidf.max() > scores_tfidf.min():
        scores_tfidf_norm = scaler.fit_transform(scores_tfidf.reshape(-1, 1)).flatten()
    else:
        scores_tfidf_norm = np.zeros_like(scores_tfidf)

    # --- FUSION HYBRIDE ---
    scores_hybrid = (alpha_als * scores_als_norm) + (alpha_tfidf * scores_tfidf_norm)

    # --- EXCLUSION DES ITEMS DÉJÀ VUS ---
    for idx in read_indices:
        scores_hybrid[idx] = -np.inf

    # --- TOP-K ---
    top_indices = scores_hybrid.argsort()[-top_k * 2:][::-1]  # Prendre 2x plus pour filtrage

    recommendations = []
    for idx in top_indices:
        item_id = idx_to_item[idx]
        if item_id not in user_history:
            recommendations.append(item_id)
            if len(recommendations) >= top_k:
                break

    # --- FALLBACK POPULARITÉ ---
    if len(recommendations) < top_k:
        for idx in popular_indices:
            item_id = idx_to_item[idx]
            if item_id not in user_history and item_id not in recommendations:
                recommendations.append(item_id)
                if len(recommendations) >= top_k:
                    break

    return recommendations[:top_k]


# ==============================================================================
# ÉVALUATION MAP@10 (CORRIGÉE)
# ==============================================================================

def evaluate_map_at_10_corrected(test_df, alpha_als, alpha_tfidf):
    """
    Calcule le MAP@10 avec la formule correcte.

    MAP@10 = moyenne des AP@10 pour tous les utilisateurs
    AP@10 = (somme des précisions aux rangs où un item pertinent apparaît) / min(10, nb_items_pertinents)
    """

    test_ground_truth = test_df.groupby('u')['i'].apply(set).to_dict()
    average_precisions = []

    valid_test_users = [u for u in test_ground_truth.keys() if u in user_to_idx]

    for user_id in tqdm(valid_test_users, desc=f"Eval α_als={alpha_als:.2f}, α_tfidf={alpha_tfidf:.2f}", leave=False):

        rec_items = recommend_hybrid_optimized(user_id, alpha_als, alpha_tfidf, top_k=10)

        if not rec_items:
            continue

        true_items = test_ground_truth[user_id]

        if not true_items:
            continue

        # Calcul AP@10
        hits = 0
        sum_precisions = 0.0

        for rank, item_id in enumerate(rec_items, start=1):
            if item_id in true_items:
                hits += 1
                precision_at_rank = hits / rank
                sum_precisions += precision_at_rank

        # Division par min(10, nombre d'items pertinents)
        ap = sum_precisions / min(len(true_items), 10)
        average_precisions.append(ap)

    return np.mean(average_precisions) if average_precisions else 0.0


# ==============================================================================
# GRID SEARCH OPTIMISÉ
# ==============================================================================

print("\n" + "=" * 80)
print("🔍 GRID SEARCH HYBRIDE (ALS + TF-IDF)")
print("=" * 80)

# Grid search intelligent: focus sur les zones prometteuses
alphas_als = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []

best_score = 0
best_config = {}

for alpha_als in alphas_als:
    alpha_tfidf = 1.0 - alpha_als

    score = evaluate_map_at_10_corrected(test_data, alpha_als, alpha_tfidf)
    results.append({
        'alpha_als': alpha_als,
        'alpha_tfidf': alpha_tfidf,
        'map@10': score
    })

    print(f"   α_ALS={alpha_als:.2f}, α_TFIDF={alpha_tfidf:.2f} → MAP@10 = {score:.5f}")

    if score > best_score:
        best_score = score
        best_config = {
            'alpha_als': alpha_als,
            'alpha_tfidf': alpha_tfidf
        }

# ==============================================================================
# RÉSULTATS FINAUX
# ==============================================================================

print("\n" + "=" * 80)
print("📊 RÉSULTATS FINAUX")
print("=" * 80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\n🏆 MEILLEUR MODÈLE:")
print(f"   Configuration: ALS={best_config['alpha_als']:.2f}, TF-IDF={best_config['alpha_tfidf']:.2f}")
print(f"   MAP@10: {best_score:.5f}")
print(f"   Baseline: {BASELINE_SCORE}")

improvement = ((best_score / BASELINE_SCORE) - 1) * 100
if best_score > BASELINE_SCORE:
    print(f"   ✅ Amélioration: +{improvement:.1f}%")
elif best_score > BASELINE_SCORE * 0.8:
    print(f"   ⚠️  Performance: {improvement:.1f}% (proche baseline)")
else:
    print(f"   ❌ Performance: {improvement:.1f}% (sous baseline)")

print("\n💡 ANALYSE:")
if best_config['alpha_als'] > 0.7:
    print("   → Le modèle collaboratif (ALS) domine: les interactions sont riches")
elif best_config['alpha_tfidf'] > 0.7:
    print("   → Le contenu (TF-IDF) domine: les données sont trop sparse pour le CF")
else:
    print("   → Hybride équilibré: combinaison optimale des deux approches")

print("\n" + "=" * 80)
print("✨ Analyse terminée !")
print("=" * 80)