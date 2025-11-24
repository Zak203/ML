import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize_matrix():
    print("🐱 Chargement de la matrice...")

    # On charge la matrice qu'on vient de créer
    with open('data_processed/R_full.pkl', 'rb') as f:
        R_full = pickle.load(f)

    n_users, n_items = R_full.shape
    print(f"📊 Dimensions : {n_users} users x {n_items} items")
    print(f"   Remplissage : {R_full.nnz} interactions ({R_full.nnz / (n_users * n_items):.2%} complet)")

    # --- FIGURE 1 : LE SPY PLOT (Vue d'ensemble) ---
    # C'est comme regarder le ciel étoilé. Chaque point noir est un emprunt.
    # On prend un échantillon (ex: les 1000 premiers users/items) pour que ça reste lisible.
    plt.figure(figsize=(10, 10))
    plt.spy(R_full[:1000, :1000], markersize=1, aspect='auto')
    plt.title("Vue d'ensemble (Zoom 1000x1000)\nChaque point = 1 livre emprunté", fontsize=14)
    plt.xlabel("Index Livres")
    plt.ylabel("Index Users")
    plt.show()

    # --- FIGURE 2 : LA HEATMAP DES "VIP" (Zoom sur l'activité) ---
    # On va chercher les utilisateurs les plus actifs et les livres les plus lus
    # pour voir s'il y a des "blocs" d'activité.

    # Somme sur les lignes (activité users) et colonnes (popularité livres)
    user_activity = np.array(R_full.sum(axis=1)).flatten()
    item_popularity = np.array(R_full.sum(axis=0)).flatten()

    # On prend les indices des Top 50
    top_user_idx = user_activity.argsort()[-50:][::-1]
    top_item_idx = item_popularity.argsort()[-50:][::-1]

    # On extrait le petit carré dense
    dense_block = R_full[top_user_idx][:, top_item_idx].toarray()

    plt.figure(figsize=(12, 8))
    sns.heatmap(dense_block, cmap="Blues", cbar=True, annot=False)
    plt.title("Zoom sur l'élite : Top 50 Users vs Top 50 Livres", fontsize=14)
    plt.xlabel("Livres Populaires (Index)")
    plt.ylabel("Gros Lecteurs (Index)")
    plt.show()


if __name__ == "__main__":
    visualize_matrix()
