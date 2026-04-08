DATA_PATH = "scripts/data/ratings.csv"
MOVIES_PATH = "scripts/data/movies.csv"
CHROMA_DIR = "chroma_db"
CHECKPOINT_DIR = "checkpoints"
COLLECTION_NAME = "movie_items"

EMBEDDING_DIM = 128
NUM_NEGATIVES = 8
BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-3

GENRE_LIST = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]
NUM_GENRES = len(GENRE_LIST)
GENRE_PADDING_IDX = NUM_GENRES
MAX_GENRE_LEN = 6
