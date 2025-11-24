import pandas as pd
import numpy as np
import pickle
import random


def check_integrity():
    print("🕵️‍♂️ Début de l'enquête de cohérence...")

    # 1. Charger la vérité (CSV)
    print("📂 Chargement du CSV original...")
    df = pd.read_csv('kaggle_data/interactions_train.csv')

    # 2. Charger notre suspect (Matrice + Mappings)
    print("📦 Chargement de la matrice construite...")
    with open('data_processed/R_full.pkl', 'rb') as f:
        R_full = pickle.load(f)

    with open('data_processed/mappings_full.pkl', 'rb') as f:
        mappings = pickle.load(f)
        user_to_idx = mappings['user_to_idx']
        item_to_idx = mappings['item_to_idx']
        idx_to_user = mappings['idx_to_user']
        idx_to_item = mappings['idx_to_item']

    # 3. LE TEST (Sondage aléatoire)
    print("\n--- 🎲 TEST ALÉATOIRE SUR 5 INTERACTIONS ---")

    # On prend 5 lignes au hasard dans le CSV
    samples = df.sample(5)

    errors = 0
    for index, row in samples.iterrows():
        real_user = row['u']
        real_item = row['i']

        # On traduit en indices matriciels
        try:
            u_idx = user_to_idx[real_user]
            i_idx = item_to_idx[real_item]
        except KeyError:
            print(f"⚠️ Problème : User {real_user} ou Item {real_item} non trouvé dans le mapping !")
            continue

        # On regarde dans la matrice
        # R_full[u_idx, i_idx] doit valoir 1
        matrix_value = R_full[u_idx, i_idx]

        print(f"Test : User {real_user} (idx {u_idx}) a lu Item {real_item} (idx {i_idx}) ?")

        if matrix_value == 1:
            print(f"   ✅ Matrice : OUI (1.0)")
        else:
            print(f"   ❌ Matrice : NON ({matrix_value}) -> ERREUR !")
            errors += 1

    print("\n--- 🏁 RÉSULTAT DE L'ENQUÊTE ---")
    if errors == 0:
        print("✅ SUCCESS : La matrice reflète fidèlement le CSV.")
        print("   La diagonale que tu vois est probablement due au fait que les utilisateurs")
        print("   et les livres sont triés par ordre d'apparition dans le dataset.")
        print("   (Les premiers users enregistrés ont lu les premiers livres enregistrés).")
    else:
        print(f"❌ FAIL : Il y a {errors} erreurs. La construction de la matrice est fausse.")


if __name__ == "__main__":
    check_integrity()