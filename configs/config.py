from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

BEST_CHECKPOINT = CHECKPOINT_DIR / "best_iou.pt"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"

MODEL_DIR = PROJECT_ROOT / "models"

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

BATCH_SIZE = 8

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

VAL_SPLIT = 0.20

NUM_WORKERS = 2

PREFETCH_FACTOR = 4

PERSISTENT_WORKERS = True

PIN_MEMORY = True

SAVE_EVERY_BATCHES = 1400

EARLY_STOPPING_PATIENCE = 3

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

NUM_CLASSES = 81

IMAGE_SIZE = (512, 512)

TORCHSCRIPT_MODEL = MODEL_DIR / "deeplabv3_traced.pt"