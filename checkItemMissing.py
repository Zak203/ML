import pandas as pd

# === A ADAPTER SI BESOIN ===
BASE_FILE = "kaggle_data/items.csv"                      # fichier de base (Kaggle)
ENRICHED_FILE = "resultats_enrichie/final/books_enriched_BASE.xlsx"  # ton fichier enrichi actuel
OUTPUT_FILE = "resultats_enrichie/final/books_enriched_FIXED.xlsx"   # nouveau fichier de sortie

ITEMS_SHEET = "Sheet1"   # nom de la feuille de ton Excel enrichi


def main():
    # 1) Charger les deux fichiers
    print(f"📘 Chargement fichier de base : {BASE_FILE}")
    df_base = pd.read_csv(BASE_FILE)

    print(f"📘 Chargement fichier enrichi : {ENRICHED_FILE}")
    df_enriched = pd.read_excel(ENRICHED_FILE, sheet_name=ITEMS_SHEET)

    # Sécurité : vérifier qu'on a bien la colonne 'i'
    if "i" not in df_base.columns:
        raise ValueError("La colonne 'i' est absente du fichier de base.")
    if "i" not in df_enriched.columns:
        raise ValueError("La colonne 'i' est absente du fichier enrichi.")

    # 2) Identifier les i manquants
    base_ids = set(df_base["i"].unique())
    enriched_ids = set(df_enriched["i"].unique())

    missing_ids = sorted(base_ids - enriched_ids)
    print(f"🔎 Nombre d'items manquants dans l'enrichi : {len(missing_ids)}")

    if not missing_ids:
        print("✅ Aucun item manquant, rien à corriger.")
        return

    # 3) Récupérer les lignes manquantes depuis le fichier de base
    df_missing = df_base[df_base["i"].isin(missing_ids)].copy()
    print("👀 Aperçu des lignes manquantes (fichier de base) :")
    print(df_missing.head())

    # 4) Créer les lignes à ajouter avec la même structure que l'enrichi
    #    - On ne garde que les colonnes communes
    #    - Les colonnes supplémentaires de l'enrichi seront remplies à NaN
    common_cols = [c for c in df_enriched.columns if c in df_missing.columns]
    extra_cols = [c for c in df_enriched.columns if c not in df_missing.columns]

    # On part des colonnes communes
    df_new_rows = df_missing[common_cols].copy()

    # On ajoute les colonnes qui existent dans l'enrichi mais pas dans le base
    for col in extra_cols:
        df_new_rows[col] = pd.NA

    # On remet les colonnes dans le même ordre que df_enriched
    df_new_rows = df_new_rows[df_enriched.columns]

    print(f"➕ Lignes à ajouter : {df_new_rows.shape[0]}")
    print("👀 Aperçu des nouvelles lignes formatées comme l'enrichi :")
    print(df_new_rows.head())

    # 5) Concaténer à la fin du tableau enrichi
    df_fixed = pd.concat([df_enriched, df_new_rows], ignore_index=True)

    # 6) Sauvegarder dans un NOUVEAU fichier Excel (pour ne pas casser l’original)
    df_fixed.to_excel(OUTPUT_FILE, index=False, sheet_name=ITEMS_SHEET)
    print(f"✅ Nouveau fichier généré : {OUTPUT_FILE}")
    print(f"📏 Nouvelle taille : {df_fixed.shape}")


if __name__ == "__main__":
    main()