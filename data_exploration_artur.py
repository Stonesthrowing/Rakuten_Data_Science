from pathlib import Path

# Projektordner finden
PROJECT_ROOT = Path(__file__).resolve().parent

# Datenordner
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATA_DIR / "image_train"
TEST_DIR = DATA_DIR / "image_test"

print("Train folder:", TRAIN_DIR)
print("Test folder:", TEST_DIR)

# Beispiel: Anzahl Bilder zählen
train_images = list(TRAIN_DIR.glob("*.jpg"))
test_images = list(TEST_DIR.glob("*.jpg"))

print("Train images:", len(train_images))
print("Test images:", len(test_images))