import os
import sys
import urllib.request
import zipfile
import csv

DATA_DIR = "scripts/data"

DATASETS = {
    "100k": {
        "url": "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "zip": "ml-100k.zip",
        "folder": "ml-100k",
        "format": "100k",
    },
    "1m": {
        "url": "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "zip": "ml-1m.zip",
        "folder": "ml-1m",
        "format": "1m",
    },
}

GENRE_NAMES_100K = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def download_and_extract(url, zip_name, folder):
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, zip_name)
    folder_path = os.path.join(DATA_DIR, folder)

    if os.path.exists(folder_path):
        print(f"Already exists: {folder_path}, skipping download.")
        return

    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)


# ── 100K ──────────────────────────────────────────────────────────────

def build_100k():
    folder = os.path.join(DATA_DIR, "ml-100k")

    # ratings.csv
    src = os.path.join(folder, "u.data")
    dst = os.path.join(DATA_DIR, "ratings.csv")
    with open(src, "r") as fin, open(dst, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["user_id", "item_id", "rating", "timestamp"])
        for line in fin:
            writer.writerow(line.strip().split("\t"))
    print(f"Saved {dst}")

    # movies.csv
    src = os.path.join(folder, "u.item")
    dst = os.path.join(DATA_DIR, "movies.csv")
    with open(src, "r", encoding="latin-1") as fin, open(dst, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["item_id", "title", "genre"])
        for line in fin:
            fields = line.strip().split("|")
            item_id = fields[0]
            title = fields[1]
            genre_flags = fields[5:]
            genres = [GENRE_NAMES_100K[i] for i, flag in enumerate(genre_flags) if flag == "1"]
            genre_str = "|".join(genres) if genres else "unknown"
            writer.writerow([item_id, title, genre_str])
    print(f"Saved {dst}")


# ── 1M ────────────────────────────────────────────────────────────────

def build_1m():
    folder = os.path.join(DATA_DIR, "ml-1m")

    # ratings.csv — format: UserID::MovieID::Rating::Timestamp
    src = os.path.join(folder, "ratings.dat")
    dst = os.path.join(DATA_DIR, "ratings.csv")
    with open(src, "r", encoding="latin-1") as fin, open(dst, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["user_id", "item_id", "rating", "timestamp"])
        for line in fin:
            parts = line.strip().split("::")
            writer.writerow(parts)
    print(f"Saved {dst}")

    # movies.csv — format: MovieID::Title::Genres (pipe-separated, already string)
    src = os.path.join(folder, "movies.dat")
    dst = os.path.join(DATA_DIR, "movies.csv")
    with open(src, "r", encoding="latin-1") as fin, open(dst, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["item_id", "title", "genre"])
        for line in fin:
            parts = line.strip().split("::")
            item_id = parts[0]
            title = parts[1]
            genre_str = parts[2]  # already "Action|Comedy|..." format
            writer.writerow([item_id, title, genre_str])
    print(f"Saved {dst}")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    version = sys.argv[1].lower() if len(sys.argv) > 1 else "1m"
    if version not in DATASETS:
        print(f"Unknown version: {version}. Choose from: {list(DATASETS.keys())}")
        sys.exit(1)

    cfg = DATASETS[version]
    download_and_extract(cfg["url"], cfg["zip"], cfg["folder"])

    if cfg["format"] == "100k":
        build_100k()
    else:
        build_1m()

    print(f"Done! MovieLens {version.upper()} ready in {DATA_DIR}/")
