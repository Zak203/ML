import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============= CONFIGURATION =================
#INPUT_FILE = "mlproject.xlsx"
INPUT_FILE = "books_enriched_sumary_Manque_Part1.xlsx"
#SHEET_NAME = "items"
SHEET_NAME = "Sheet1"
ISBN_COL = "ISBN Valid"
MAX_WORKERS = 20      # plus de threads en parallèle
TEST_ROWS = None        # mets None pour traiter tout
TIMEOUT = 5          # Timeout API en secondes

# ============= FONCTIONS UTILITAIRES =================

def extract_isbn_list(raw_isbn):
    """
    Prend une cellule contenant 1 ou plusieurs ISBN (séparés par ; ou ,)
    et retourne une liste d’ISBN propres.
    """
    if not isinstance(raw_isbn, str):
        return []
    cleaned = raw_isbn.replace(";", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    # on enlève les doublons en gardant l’ordre
    seen = set()
    result = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def empty_record():
    """Record vide avec les champs que l'on veut remplir."""
    return {
        "summary": None,
        "published_year": None,
        "page_count": None,
        "language": None,
        #"average_rating": None,
        #"popularity": None,
    }


def merge_records(base, extra):
    """
    Fusionne deux dicts de metadata :
    - garde base[x] si déjà présent
    - utilise extra[x] uniquement si base[x] est None
    """
    if not extra:
        return base
    for k, v in extra.items():
        if k in base and base[k] is None and v is not None:
            base[k] = v
    return base


def missing_core_fields(m):
    """True si au moins un des champs importants est encore None."""
    return any(m[k] is None for k in ["summary", "published_year", "page_count", "language"])


def missing_summary(m):
    """True si le résumé est manquant."""
    return m.get("summary") is None


# ============= GOOGLE BOOKS =================

def fetch_google_books(isbn):
    """Retourne un dict avec résumé, année, pages, langue, rating, popularity depuis Google Books."""
    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()

        items = data.get("items")
        if not items:
            return None

        info = items[0].get("volumeInfo", {})

        return {
            "summary": info.get("description"),
            "published_year": info.get("publishedDate"),
            "page_count": info.get("pageCount"),
            "language": info.get("language"),
            #"average_rating": info.get("averageRating"),
            #"popularity": info.get("ratingsCount"),
        }
    except Exception:
        return None


# ============= OPENLIBRARY PAR ISBN =================

def fetch_openlibrary_isbn(isbn):
    """Récupère des infos OpenLibrary via l'API books?bibkeys=ISBN:xxx."""
    url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&jscmd=details&format=json"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()

        key = f"ISBN:{isbn}"
        if key not in data:
            return None

        details = data[key].get("details", {})

        # ----- description & notes -----
        desc = details.get("description")
        if isinstance(desc, dict):
            desc = desc.get("value")

        if not isinstance(desc, str) or not desc.strip():
            # fallback sur notes si description vide
            notes = details.get("notes")
            if isinstance(notes, dict):
                notes = notes.get("value")
            if isinstance(notes, str) and notes.strip():
                desc = notes.strip()

        if not isinstance(desc, str):
            desc = None

        # ----- langue -----
        lang = None
        langs = details.get("languages")
        if isinstance(langs, list) and langs:
            lang = langs[0].get("key")  # ex: "/languages/fre"

        # ----- popularité (borrow_count) -----
        borrow_count = None
        try:
            works = details.get("works") or []
            if works:
                work_key = works[0].get("key")
                if work_key:
                    borrow_url = f"https://openlibrary.org{work_key}.json"
                    br = requests.get(borrow_url, timeout=TIMEOUT)
                    br.raise_for_status()
                    borrow_data = br.json()
                    borrow_count = borrow_data.get("borrow_count")
        except Exception:
            pass

        return {
            "summary": desc,
            "published_year": details.get("publish_date"),
            "page_count": details.get("number_of_pages"),
            "language": lang,
            #"average_rating": None,   # pas dispo dans cette API
            #"popularity": borrow_count,
        }
    except Exception:
        return None


# ============= OPENLIBRARY PAR TITRE / AUTEUR =================

def fetch_openlibrary_title_author(title, author=None):
    """
    Fallback : recherche OpenLibrary par titre (+ auteur si dispo),
    puis va chercher la work pour récupérer description / année / langue.
    """
    if not isinstance(title, str) or not title.strip():
        return None

    params = {
        "title": title,
        "limit": 1,
    }
    if isinstance(author, str) and author.strip():
        params["author"] = author

    try:
        r = requests.get("https://openlibrary.org/search.json", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        docs = data.get("docs") or []
        if not docs:
            return None

        doc = docs[0]
        work_key = doc.get("key")  # ex: "/works/OL123456W"
        first_publish_year = doc.get("first_publish_year")

        desc = None
        lang = None

        # Aller chercher la work pour description
        if work_key:
            try:
                wr = requests.get(f"https://openlibrary.org{work_key}.json", timeout=TIMEOUT)
                wr.raise_for_status()
                wdata = wr.json()
                d = wdata.get("description")
                if isinstance(d, dict):
                    desc = d.get("value")
                elif isinstance(d, str):
                    desc = d

                langs = wdata.get("languages")
                if isinstance(langs, list) and langs:
                    lang = langs[0].get("key")
            except Exception:
                pass

        return {
            "summary": desc,
            "published_year": first_publish_year,
            "page_count": None,
            "language": lang,
            #"average_rating": None,
            #"popularity": None,
        }
    except Exception:
        return None


# ============= PIPELINE PRINCIPAL =================

def main():
    # ---- 1. Lire sheet items ----
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    # ---- 2. Limiter pour test ----
    if TEST_ROWS is not None:
        df = df.head(TEST_ROWS)

    # ---- 3. Extraire liste d’ISBN par ligne ----
    df["isbn_list"] = df[ISBN_COL].apply(extract_isbn_list)
    df["first_isbn"] = df["isbn_list"].apply(lambda lst: lst[0] if lst else None)

    # ---- 4. Construire mapping ISBN -> titre / auteur (pour fallback titre+auteur) ----
    title_col = "Title"
    author_col = "Author"

    isbn_to_title = {}
    isbn_to_author = {}

    for _, row in df.iterrows():
        title = row.get(title_col)
        author = row.get(author_col)
        for isbn in row["isbn_list"]:
            if isbn not in isbn_to_title:
                isbn_to_title[isbn] = title
            if isbn not in isbn_to_author:
                isbn_to_author[isbn] = author

    # ---- 5. ISBN uniques à appeler dans les APIs ----
    unique_isbns = sorted({isbn for lst in df["isbn_list"] for isbn in lst})
    print(f"🔍 ISBN uniques à traiter : {len(unique_isbns)}")

    isbn_to_data = {}

    def process_single_isbn(isbn):
        """Google -> OL ISBN -> OL titre/auteur pour UN ISBN donné."""
        meta = empty_record()

        # 1) Google Books
        gb = fetch_google_books(isbn)
        if gb:
            meta = merge_records(meta, gb)

        # Si Google a déjà tout, inutile d'aller voir OpenLibrary
        if not missing_core_fields(meta):
            return isbn, meta

        # 2) OpenLibrary via ISBN si infos manquantes
        ol_isbn = fetch_openlibrary_isbn(isbn)
        if ol_isbn:
            meta = merge_records(meta, ol_isbn)

        # Si on a déjà un résumé, ne pas lancer la recherche par titre/auteur (très lente)
        if not missing_summary(meta):
            return isbn, meta

        # 3) OpenLibrary via titre + auteur si toujours incomplet
        title = isbn_to_title.get(isbn)
        author = isbn_to_author.get(isbn)
        ol_ta = fetch_openlibrary_title_author(title, author)
        if ol_ta:
            meta = merge_records(meta, ol_ta)

        return isbn, meta

    # ---- 6. Appels API en parallèle pour tous les ISBN uniques ----
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_isbn, isbn): isbn for isbn in unique_isbns}

        for i, future in enumerate(as_completed(futures), start=1):
            isbn = futures[future]
            isbn_ret, result = future.result()
            isbn_to_data[isbn_ret] = result
            print(f"Progression ISBN : {i}/{len(futures)}")

    # ---- 7. Construire les métadonnées par LIGNE (en utilisant tous les ISBN de la ligne) ----
    def build_row_meta(isbn_list):
        meta = empty_record()
        for isbn in isbn_list:
            data = isbn_to_data.get(isbn)
            if data:
                meta = merge_records(meta, data)
            # si on a tout, on peut s'arrêter
            if not missing_core_fields(meta):
                break
        return meta

    row_meta_series = df["isbn_list"].apply(build_row_meta)
    meta_df = row_meta_series.apply(pd.Series)

    # meta_df a les colonnes summary, published_year, page_count, language, average_rating, popularity
    for col in meta_df.columns:
        df[col] = meta_df[col]

    # ---- 8. Sauvegarde test ----
    df.to_excel("books_enriched_sumary_Manqu_part1_result.xlsx", index=False)
    print("✅ Test terminé → resultat : books_enriched_TEST.xlsx")


if __name__ == "__main__":
    main()