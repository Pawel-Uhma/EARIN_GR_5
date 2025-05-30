import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))

DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "processed")
CSV_FILE   = os.path.join(DATA_DIR, "train.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")

IMG_SIZE      = 64
BATCH_SIZE    = 32
NUM_EPOCHS    = 5
LEARNING_RATE = 1e-4
VAL_SPLIT     = 0.2
RANDOM_SEED   = 42
NUM_CLASSES   = 3  # YOUNG, MIDDLE, OLD

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)