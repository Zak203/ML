import pandas as pd
import numpy as np
import sys
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, MinMaxScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INTERACTIONS_PATH = 'kaggle_data/interactions_train.csv'
ITEMS_PATH = 'kaggle_data/items.csv'
BASELINE_SCORE = 0.1452

print("=" * 80)
print("🚀 CF + TF-IDF MULTI-LEVEL SUBJECTS (Approche Optimale)")
print("=" * 80)

# ==============================================================================
# CHARGEMENT
# ==============================================================================

print("\n📂 Chargement...")
try:
    interactions_df = pd.read_csv(INTERACTIONS_PATH)
    items_df = pd.read_csv(ITEMS_PATH)
except FileNotFoundError as e:
    print(f"❌ {e}")
    sys.exit(1)

if 'user_id' in interactions_df.columns:
    interactions_df = interactions_df.rename(columns={
        'user_id': 'u', 'item_id': 'i', 'timestamp': 't'
    })

print(f"   Interactions: {len(interactions_df):,}")
print(f"   Users: {interactions_df['u'].nunique():,}")
print(f"   Items: {interactions_df['i'].nunique():,}")

# ==============================================================================
# PRÉPARATION DES MÉTADONNÉES ENRICHIES
# ==============================================================================

print("\n📝 Préparation des métadonnées...")

items_df['Title'] = items_df['Title'].fillna('').str.lower()
items_df['Author'] = items_df['Author'].fillna('').str.lower()
items_df['Subjects'] = items_df['Subjects'].fillna('')

# Analyser les subjects
subjects_sample = items_df['Subjects'].head(10)
print("\n📚 Exemples de Subjects:")
for i, subj in enumerate(subjects_sample, 1):
    if subj:
        print(f"   {i}. {subj[:100]}...")

# Statistiques
n_with_subjects = (items_df['Subjects'] != '').sum()
print(f"\n   Items avec Subjects: {n_with_subjects:,} ({n_with_subjects / len(items_df) * 100:.1f}%)")


# Créer une "soupe enrichie" qui capture TOUS les aspects
def create_enriched_soup(row):
    """
    Crée une soupe textuelle qui capture:
    - Titre (2x pour importance)
    - Auteur (1x)
    - TOUS les subjects (5x pour maximum d'importance)
    """
    title = row['Title']
    author = row['Author']
    subjects = row['Subjects']

    # Nettoyer les subjects: remplacer ; par espace, enlever ponctuation spéciale
    subjects_cleaned = subjects.replace(';', ' ').replace('--', ' ').replace('[', '').replace(']', '')

    # Composition: Titre 2x, Auteur 1x, Subjects 5x
    soup = f"{title} {title} {author} " + (f"{subjects_cleaned} " * 5)

    return soup


items_df['content_soup'] = items_df.apply(create_enriched_soup, axis=1)

print(f"\n   Exemple de soupe enrichie:")
example_soup = items_df.iloc[0]['content_soup']
print(f"   '{example_soup[:200]}...'")

# ==============================================================================
# SPLIT TEMPOREL
# ==============================================================================

print("\n⏱️  Split temporel 80/20...")

interactions_df = interactions_df.sort_values(['u', 't'])
interactions_df["pct_rank"] = interactions_df.groupby("u")["t"].rank(pct=True, method='first')
train_data = interactions_df[interactions_df["pct_rank"] <= 0.8].copy()
test_data = interactions_df[interactions_df["pct_rank"] > 0.8].copy()

users_in_train = set(train_data['u'].unique())
test_data = test_data[test_data['u'].isin(users_in_train)]

print(f"   Train: {len(train_data):,}")
print(f"   Test: {len(test_data):,}")

# ==============================================================================
# MATRICE R_train
# ==============================================================================

print("\n🗺️  Création de R_train...")

unique_users = sorted(train_data['u'].unique())
unique_items = sorted(train_data['i'].unique())

user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
idx_to_item = {idx: i for i, idx in item_to_idx.items()}

train_rows = train_data['u'].map(user_to_idx).values
train_cols = train_data['i'].map(item_to_idx).values

R_train = csr_matrix(
    (np.ones(len(train_data)), (train_rows, train_cols)),
    shape=(len(unique_users), len(unique_items))
)

print(f"   Matrice: {R_train.shape}")

user_history_map = train_data.groupby('u')['i'].apply(set).to_dict()

# Popularité
item_popularity = np.array(R_train.sum(axis=0)).flatten()
popular_indices = item_popularity.argsort()[-100:][::-1]

# ==============================================================================
# MODÈLE 1: ITEM-ITEM CF CLASSIQUE
# ==============================================================================

print("\n🔹 Calcul Item-Item CF (Cosine Similarity)...")

item_item_sim_cf = cosine_similarity(R_train.T, dense_output=True)
np.fill_diagonal(item_item_sim_cf, 0)

print(f"   Similarité CF: {item_item_sim_cf.shape}")

# ==============================================================================
# MODÈLE 2: TF-IDF SUR SUBJECTS ENRICHIS
# ==============================================================================

print("\n🔹 Calcul TF-IDF multi-level sur Subjects...")


# TF-IDF avec tokenizer custom pour gérer les subjects multiples
def custom_tokenizer(text):
    """
    Tokenizer qui:
    1. Supprime la ponctuation spéciale
    2. Split sur espaces
    3. Garde les tokens de 2+ caractères
    """
    # Remplacer ponctuation par espaces
    for char in [';', '--', '(', ')', '[', ']', ',']:
        text = text.replace(char, ' ')

    # Split et filtrer
    tokens = [t.strip().lower() for t in text.split() if len(t.strip()) > 1]
    return tokens


tfidf = TfidfVectorizer(
    tokenizer=custom_tokenizer,
    min_df=2,  # Ignorer les termes ultra-rares
    max_df=0.5,  # Ignorer les termes ultra-communs
    ngram_range=(1, 3),  # Uni/bi/tri-grammes pour capturer "langue étrangère"
    lowercase=True,
    strip_accents='unicode'
)

tfidf_matrix = tfidf.fit_transform(items_df['content_soup'])

print(f"   TF-IDF: {tfidf_matrix.shape}")
print(f"   Features extraites: {len(tfidf.get_feature_names_out()):,}")

# Quelques features exemples
features_sample = tfidf.get_feature_names_out()[:20]
print(f"   Exemples de features: {', '.join(features_sample)}")

# Calcul de la similarité TF-IDF
print("\n   Calcul de la similarité cosinus TF-IDF...")
item_sim_tfidf = linear_kernel(tfidf_matrix, tfidf_matrix)
np.fill_diagonal(item_sim_tfidf, 0)

print(f"   Similarité TF-IDF: {item_sim_tfidf.shape}")

# ==============================================================================
# ALIGNEMENT CF <-> TF-IDF
# ==============================================================================

print("\n🔀 Alignement des matrices...")

# Créer le mapping item_id -> index dans items_df
item_id_to_content_idx = pd.Series(items_df.index, index=items_df['i']).to_dict()

# Pour chaque item dans l'espace CF, trouver son index dans l'espace TF-IDF
aligned_content_indices = []
valid_cf_indices = []

for cf_idx in range(len(unique_items)):
    item_id = idx_to_item[cf_idx]
    if item_id in item_id_to_content_idx:
        content_idx = item_id_to_content_idx[item_id]
        aligned_content_indices.append(content_idx)
        valid_cf_indices.append(cf_idx)

print(f"   Items alignés: {len(aligned_content_indices):,} / {len(unique_items):,}")

# Extraire les sous-matrices alignées
item_sim_cf_aligned = item_sim_tfidf[np.ix_(aligned_content_indices, aligned_content_indices)]

print(f"   Matrice TF-IDF alignée: {item_sim_cf_aligned.shape}")


# ==============================================================================
# RECOMMANDATION HYBRIDE
# ==============================================================================

def recommend_hybrid(user_id, alpha_cf, alpha_tfidf, top_k=10):
    """
    Recommandation hybride CF + TF-IDF
    """

    if user_id not in user_to_idx:
        return []

    u_idx = user_to_idx[user_id]
    user_history = user_history_map.get(user_id, set())

    # Items lus
    user_row = R_train.getrow(u_idx)
    read_indices = user_row.indices

    if len(read_indices) == 0:
        return [idx_to_item[idx] for idx in popular_indices[:top_k]]

    # --- SCORES CF ---
    scores_cf = np.zeros(len(unique_items))
    for idx in read_indices:
        scores_cf += item_item_sim_cf[idx]

    # --- SCORES TF-IDF ---
    scores_tfidf = np.zeros(len(unique_items))

    # Pour chaque item lu, ajouter les scores de similarité TF-IDF
    for idx in read_indices:
        # Trouver l'index dans l'espace aligné
        if idx in valid_cf_indices:
            aligned_idx = valid_cf_indices.index(idx)

            # Récupérer la ligne de similarité
            for j, cf_idx in enumerate(valid_cf_indices):
                scores_tfidf[cf_idx] += item_sim_cf_aligned[aligned_idx, j]

    # --- NORMALISATION ---
    scaler = MinMaxScaler()

    if scores_cf.max() > scores_cf.min():
        scores_cf_norm = scaler.fit_transform(scores_cf.reshape(-1, 1)).flatten()
    else:
        scores_cf_norm = np.zeros_like(scores_cf)

    if scores_tfidf.max() > scores_tfidf.min():
        scores_tfidf_norm = scaler.fit_transform(scores_tfidf.reshape(-1, 1)).flatten()
    else:
        scores_tfidf_norm = np.zeros_like(scores_tfidf)

    # --- FUSION ---
    scores_hybrid = alpha_cf * scores_cf_norm + alpha_tfidf * scores_tfidf_norm

    # Exclure vus
    scores_hybrid[read_indices] = -np.inf

    # Top-K
    top_indices = scores_hybrid.argsort()[-top_k * 2:][::-1]

    recommendations = []
    for idx in top_indices:
        item_id = idx_to_item[idx]
        if item_id not in user_history:
            recommendations.append(item_id)
            if len(recommendations) >= top_k:
                break

    # Fallback
    if len(recommendations) < top_k:
        for idx in popular_indices:
            item_id = idx_to_item[idx]
            if item_id not in user_history and item_id not in recommendations:
                recommendations.append(item_id)
                if len(recommendations) >= top_k:
                    break

    return recommendations[:top_k]


# ==============================================================================
# ÉVALUATION
# ==============================================================================

def evaluate_hybrid(test_df, alpha_cf, alpha_tfidf):
    """
    Évalue le modèle hybride
    """

    test_truth = test_df.groupby('u')['i'].apply(set).to_dict()
    aps = []

    valid_users = [u for u in test_truth if u in user_to_idx]

    for user_id in tqdm(valid_users, desc=f"CF={alpha_cf:.2f} TFIDF={alpha_tfidf:.2f}", leave=False):

        rec = recommend_hybrid(user_id, alpha_cf, alpha_tfidf)

        if not rec:
            continue

        true = test_truth[user_id]
        if not true:
            continue

        hits = 0
        sum_prec = 0.0

        for rank, item in enumerate(rec, 1):
            if item in true:
                hits += 1
                sum_prec += hits / rank

        ap = sum_prec / min(len(true), 10)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


# ==============================================================================
# GRID SEARCH
# ==============================================================================

print("\n" + "=" * 80)
print("🔍 GRID SEARCH: Optimisation CF vs TF-IDF Multi-Level")
print("=" * 80)

configs = [
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
]

results = []
best_score = 0
best_config = None

for alpha_cf, alpha_tfidf in configs:

    score = evaluate_hybrid(test_data, alpha_cf, alpha_tfidf)

    results.append({
        'alpha_CF': alpha_cf,
        'alpha_TFIDF': alpha_tfidf,
        'MAP@10': score
    })

    print(f"   CF={alpha_cf:.1f}, TFIDF={alpha_tfidf:.1f} → MAP@10 = {score:.5f}")

    if score > best_score:
        best_score = score
        best_config = (alpha_cf, alpha_tfidf)

# ==============================================================================
# RÉSULTATS FINAUX
# ==============================================================================

print("\n" + "=" * 80)
print("📊 RÉSULTATS FINAUX")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('MAP@10', ascending=False)

print(results_df.to_string(index=False))

print(f"\n🏆 MEILLEUR MODÈLE:")
print(f"   CF={best_config[0]:.2f}, TF-IDF={best_config[1]:.2f}")
print(f"   MAP@10: {best_score:.5f}")
print(f"   Baseline prof: {BASELINE_SCORE}")
print(f"   Ratio: {best_score / BASELINE_SCORE * 100:.1f}%")

if best_score > BASELINE_SCORE * 0.9:
    print("\n🎉 EXCELLENT! Niveau du prof atteint!")
elif best_score > BASELINE_SCORE * 0.6:
    print("\n✅ TRÈS BON! On se rapproche")
    print("\n💡 Pour aller plus loin:")
    print("   → Installer ALS: pip install implicit")
    print("   → Tester avec ngram_range=(1,4) pour capturer 'langue étrangère enseignement'")
elif best_score > 0.05:
    print("\n⚠️  PROGRÈS SIGNIFICATIF mais pas encore suffisant")
    print("\n🔧 Actions suggérées:")
    print("   1. Installer implicit (ALS meilleur que cosine similarity)")
    print("   2. Vérifier si le split 80/20 est le même que le prof")
    print("   3. Essayer leave-one-out evaluation")
else:
    print("\n❌ PROBLÈME PERSISTANT")
    print("\n🤔 Hypothèses:")
    print("   1. Le prof a peut-être filtré les users avec <5 interactions")
    print("   2. Métrique d'évaluation différente (Recall@10 au lieu de MAP@10 ?)")
    print("   3. Split différent (global 80/20 au lieu de par-user)")

print("\n🚀 PROCHAINE ÉTAPE:")
print("   Si score < 0.10: Installer implicit et essayer ALS")
print("   Commande: pip install implicit")

print("=" * 80)