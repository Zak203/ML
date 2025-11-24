import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def generate_knn_submission():
    print("🐱 RecoCat: Chargement des données...")

    # 1. Charger la matrice COMPLÈTE (R_full) et les mappings
    try:
        with open('data_processed/R_full.pkl', 'rb') as f:
            R_full = pickle.load(f)
        with open('data_processed/mappings_full.pkl', 'rb') as f:
            mappings = pickle.load(f)
    except FileNotFoundError:
        print("❌ Fichiers manquants ! Lance d'abord '01_build_full_matrix.py'.")
        return

    user_to_idx = mappings['user_to_idx']
    idx_to_item = mappings['idx_to_item']

    # 2. Charger la liste des utilisateurs pour qui on doit prédire (sample_submission)
    sample_sub = pd.read_csv('kaggle_data/sample_submission.csv')  # Vérifie le chemin !
    target_users = sample_sub['user_id'].unique()

    print(f"🎯 Objectif : Générer des recommandations pour {len(target_users)} utilisateurs.")

    # 3. Entraîner le modèle KNN
    # Metric='cosine' est parfait pour des 0 et 1 (feedback implicite)
    # n_neighbors=50 : On regarde les 50 voisins les plus proches pour avoir assez de livres candidats
    print("🧠 Entraînement du modèle KNN (User-User)...")
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=50, n_jobs=-1)
    model_knn.fit(R_full)

    # 4. Calculer les livres populaires (Roue de secours / Fallback)
    # Si un user n'a pas de voisins ou pas assez de suggestions, on comble avec les best-sellers
    item_popularity = np.array(R_full.sum(axis=0)).flatten()
    popular_indices = item_popularity.argsort()[-10:][::-1]
    popular_items_ids = [str(idx_to_item[i]) for i in popular_indices]

    predictions = []

    print("🔮 Génération des prédictions...")
    # On boucle sur chaque utilisateur demandé
    for user_id in tqdm(target_users):

        # A. Trouver l'index matrice de l'utilisateur
        if user_id not in user_to_idx:
            # Cold Start total (User inconnu) -> On recommande les populaires
            rec_string = " ".join(popular_items_ids)
            predictions.append({'user_id': user_id, 'recommendation': rec_string})
            continue

        u_idx = user_to_idx[user_id]

        # B. Trouver les voisins (distances et indices)
        # On demande k+1 car le voisin le plus proche est l'utilisateur lui-même (distance 0)
        distances, neighbor_indices = model_knn.kneighbors(R_full[u_idx], return_distance=True)

        neighbor_indices = neighbor_indices.flatten()
        # On enlève le premier (c'est l'utilisateur lui-même)
        neighbor_indices = neighbor_indices[1:]

        # C. Agréger les livres des voisins
        # On regarde ce que les voisins ont lu
        candidate_items = {}

        for neighbor_idx in neighbor_indices:
            # Récupérer les livres lus par ce voisin
            # (row du voisin dans R_full)
            neighbor_books = R_full[neighbor_idx].indices

            for book_idx in neighbor_books:
                # On compte combien de voisins ont lu ce livre
                candidate_items[book_idx] = candidate_items.get(book_idx, 0) + 1

        # D. Filtrer ce que l'utilisateur a DÉJÀ lu
        user_history = R_full[u_idx].indices
        for seen_book in user_history:
            if seen_book in candidate_items:
                del candidate_items[seen_book]

        # E. Trier par fréquence (les livres les plus populaires chez les voisins)
        # sorted renvoie une liste de tuples (book_idx, count)
        sorted_candidates = sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)

        # F. Prendre le Top 10
        top_10_indices = [x[0] for x in sorted_candidates[:10]]

        # G. Convertir en vrais IDs (pour le CSV)
        top_10_ids = [str(idx_to_item[i]) for i in top_10_indices]

        # H. Fallback (si on a moins de 10 livres)
        if len(top_10_ids) < 10:
            for pop_item in popular_items_ids:
                if pop_item not in top_10_ids and len(top_10_ids) < 10:
                    # (Note: idéalement faudrait vérifier si déjà lu, mais pour le fallback rapide c'est ok)
                    top_10_ids.append(pop_item)

        # Format string "ID1 ID2 ID3..."
        rec_string = " ".join(top_10_ids)
        predictions.append({'user_id': user_id, 'recommendation': rec_string})

    # 5. Créer le CSV final
    print("💾 Sauvegarde de 'submission_knn.csv'...")
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv('submission_knn.csv', index=False)

    print("🎉 Terminé ! Fichier prêt à être envoyé.")


if __name__ == "__main__":
    generate_knn_submission()