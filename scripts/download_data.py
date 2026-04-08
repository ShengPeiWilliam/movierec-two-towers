import os
import urllib.request
import zipfile
import csv

DATA_DIR = "data"
ZIP_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading MovieLens 100K...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    os.remove(ZIP_PATH)


def build_ratings_csv():
    """u.data → data/ratings.csv (user_id, item_id, rating, timestamp)"""
    src = os.path.join(DATA_DIR, "ml-100k", "u.data")
    dst = os.path.join(DATA_DIR, "ratings.csv")

    with open(src, "r") as fin, open(dst, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["user_id", "item_id", "rating", "timestamp"])
        for line in fin:
            parts = line.strip().split("\t")
            writer.writerow(parts)

    print(f"Saved {dst}")


def build_movies_csv():
    """u.item → data/movies.csv (item_id, title, genre)"""
    src = os.path.join(DATA_DIR, "ml-100k", "u.item")
    dst = os.path.join(DATA_DIR, "movies.csv")

    with open(src, "r", encoding="latin-1") as fin, open(dst, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["item_id", "title", "genre"])
        for line in fin:
            fields = line.strip().split("|")
            item_id = fields[0]
            title = fields[1]
            genre_flags = fields[5:]
            genres = [GENRE_NAMES[i] for i, flag in enumerate(genre_flags) if flag == "1"]
            genre_str = "|".join(genres) if genres else "unknown"
            writer.writerow([item_id, title, genre_str])

    print(f"Saved {dst}")


if __name__ == "__main__":
    download_and_extract()
    build_ratings_csv()
    build_movies_csv()
    print("Done!")
