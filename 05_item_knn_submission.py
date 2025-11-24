import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import os


def generate_item_knn_submission():
    print("🐱 RecoCat: Chargement des données (Mode Item-Item)...")

    # 1. Charger la matrice et les mappings
    try:
        with open('data_processed/R_full.pkl', 'rb') as f:
            R_full = pickle.load(f)
        with open('data_processed/mappings_full.pkl', 'rb') as f:
            mappings = pickle.load(f)
    except FileNotFoundError:
        print("❌ Fichiers manquants ! Lance '01_build_full_matrix.py' d'abord.")
        return

    user_to_idx = mappings['user_to_idx']
    idx_to_item = mappings['idx_to_item']

    # --- CORRECTION ICI : ON PREND TOUS LES USERS CONNUS ---
    # Au lieu de lire sample_submission, on prend la liste de tous nos users mappés
    target_users = list(user_to_idx.keys())
    # On peut les trier pour que ce soit propre
    target_users.sort()

    print(f"🎯 Objectif : Générer des recommandations pour TOUS les {len(target_users)} utilisateurs connus.")

    # --- 2. ENTRAÎNEMENT SUR LA TRANSPOSÉE (R.T) ---
    print("🧠 Entraînement du KNN sur les ARTICLES (Item-Item)...")

    # On transpose : Les lignes deviennent les Livres
    # On utilise CSR pour la rapidité
    R_items = R_full.T.tocsr()

    # KNN Item-Item
    # n_neighbors=20 est un bon compromis vitesse/qualité
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model_knn.fit(R_items)

    # --- 3. PRÉPARATION (ROUE DE SECOURS) ---
    print("⚙️ Calcul des livres populaires (Fallback)...")
    item_popularity = np.array(R_full.sum(axis=0)).flatten()
    popular_indices = item_popularity.argsort()[-10:][::-1]

    predictions = []

    print("🔮 Génération des prédictions...")

    # Pour chaque utilisateur connu
    for user_id in tqdm(target_users):

        u_idx = user_to_idx[user_id]

        # B. Récupérer l'historique (indices des livres lus)
        user_history_indices = R_full[u_idx].indices

        if len(user_history_indices) == 0:
            # User connu mais sans historique (ça peut arriver après dédoublonnage agressif)
            top_10_ids = [str(idx_to_item[i]) for i in popular_indices]
            predictions.append({'user_id': user_id, 'recommendation': " ".join(top_10_ids)})
            continue

        # C. CUMUL DES SCORES (Item-Item Logic)
        candidate_scores = {}

        # Pour chaque livre lu par l'utilisateur
        for item_idx in user_history_indices:

            # Trouver les 20 livres les plus proches de CE livre
            distances, neighbors = model_knn.kneighbors(R_items[item_idx], n_neighbors=21)

            distances = distances.flatten()
            neighbors = neighbors.flatten()

            # On ignore le premier (le livre lui-même)
            for i in range(1, len(neighbors)):
                neighbor_idx = neighbors[i]
                dist = distances[i]
                similarity = 1.0 - dist

                # Accumulation des scores
                if neighbor_idx not in candidate_scores:
                    candidate_scores[neighbor_idx] = 0.0
                candidate_scores[neighbor_idx] += similarity

        # D. FILTRER (Ne pas recommander ce qu'il a déjà lu)
        for seen_item in user_history_indices:
            if seen_item in candidate_scores:
                del candidate_scores[seen_item]

        # E. TRIER ET PRENDRE LE TOP 10
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_10_indices = [x[0] for x in sorted_candidates[:10]]

        # F. FALLBACK POPULAIRE
        if len(top_10_indices) < 10:
            for pop_idx in popular_indices:
                if len(top_10_indices) >= 10:
                    break
                if pop_idx not in top_10_indices and pop_idx not in user_history_indices:
                    top_10_indices.append(pop_idx)

        # Formatage final
        top_10_ids = [str(idx_to_item[i]) for i in top_10_indices]
        predictions.append({'user_id': user_id, 'recommendation': " ".join(top_10_ids)})

    # 4. SAUVEGARDE
    print("💾 Sauvegarde de 'submission_item_knn_FULL.csv'...")
    sub_df = pd.DataFrame(predictions)
    sub_df.to_csv('submission_item_knn_FULL.csv', index=False)
    print("🎉 Fichier généré avec succès ! Il contient tous tes utilisateurs.")


if __name__ == "__main__":
    generate_item_knn_submission()