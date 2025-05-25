import os

DATA_DIR = "../data"
RAW_DIR = os.path.join(DATA_DIR, "Train")  # folder with .jpg images
CSV_FILE = os.path.join(DATA_DIR, "train.csv")  # annotations CSV
OUTPUT_DIR = "outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")


IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42
NUM_CLASSES = 3  # YOUNG, MIDDLE, OLD

os.makedirs(MODEL_DIR, exist_ok=True)

