import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import os


def build_full_interaction_matrix():
    print("🐱 RecoCat: Chargement du dataset complet...")

    # 1. CHARGEMENT
    # Adapte le chemin si ton fichier est ailleurs
    csv_path = 'kaggle_data/interactions_train.csv'

    if not os.path.exists(csv_path):
        print(f"❌ ERREUR : Le fichier {csv_path} est introuvable !")
        return

    df = pd.read_csv(csv_path)
    print(f"📊 Données chargées : {len(df)} interactions.")
    print(f"   Colonnes : {list(df.columns)}")

    # ...
    df = pd.read_csv(csv_path)
    print(f"📊 Données chargées : {len(df)} interactions.")

    # --- AJOUT ICI : NETTOYAGE DES DOUBLONS ---
    print("🧹 Suppression des doublons (User-Item)...")
    initial_len = len(df)
    # On garde la première interaction, on vire les suivantes
    df = df.drop_duplicates(subset=['u', 'i'])
    print(f"   Interactions réduites de {initial_len} à {len(df)} (Doublons retirés).")
    # ------------------------------------------

    # 2. MAPPING (IDs -> Index 0, 1, 2...)
    # On prend TOUS les users et TOUS les items uniques du fichier
    unique_users = df['u'].unique()
    unique_items = df['i'].unique()

    n_users = len(unique_users)
    n_items = len(unique_items)

    print(f"🌍 Univers : {n_users} utilisateurs uniques x {n_items} livres uniques.")

    print(f"🌍 Univers : {n_users} utilisateurs uniques x {n_items} livres uniques.")

    # --- CORRECTION ICI ---
    # On utilise 'idx' pour le compteur (0, 1, 2...) et 'u'/'i' pour l'ID réel
    user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
    item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
    # ----------------------

    # Traducteurs inverses (pour retrouver les vrais noms à la fin)
    idx_to_user = {idx: u for u, idx in user_to_idx.items()}
    idx_to_item = {idx: i for i, idx in item_to_idx.items()}

    # 3. CONSTRUCTION DE LA MATRICE
    print("🏗️ Construction de la matrice User-Item complète...")

    # Conversion des colonnes en index
    rows = df['u'].map(user_to_idx).values
    cols = df['i'].map(item_to_idx).values

    # Interaction implicite = 1
    data = np.ones(len(df))

    # Matrice Sparse (Lignes=Users, Colonnes=Items)
    R_full = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    print(f"✅ Matrice construite ! Shape : {R_full.shape}")
    print(f"   Densité : {R_full.nnz / (n_users * n_items):.5%}")

    # 4. SAUVEGARDE (Pickle)
    # On crée un dossier pour ranger les objets Python
    output_dir = 'data_processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"💾 Sauvegarde dans ./{output_dir} ...")

    # Sauvegarde de la matrice
    with open(f'{output_dir}/R_full.pkl', 'wb') as f:
        pickle.dump(R_full, f)

    # Sauvegarde des mappings
    mappings = {
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item
    }
    with open(f'{output_dir}/mappings_full.pkl', 'wb') as f:
        pickle.dump(mappings, f)

    print("🎉 Terminé ! Ta matrice de base est prête.")


if __name__ == "__main__":
    build_full_interaction_matrix()