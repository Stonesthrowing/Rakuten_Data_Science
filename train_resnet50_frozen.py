# train_resnet50_frozen.py

import os
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# ============================================================
# 0. Reproducibility
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 1. Configuration
# ============================================================

N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
tf.config.threading.set_intra_op_parallelism_threads(N_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(2)

DATA_ROOT = Path(os.environ["RAKUTEN_DATA_DIR"])
OUTPUT_ROOT = Path(os.environ["RAKUTEN_OUTPUT_DIR"])

RUN_NAME = "resnet50_frozen"
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_CSV = DATA_ROOT / "X_train.csv"
Y_TRAIN_CSV = DATA_ROOT / "Y_train.csv"
IMAGE_DIR = DATA_ROOT / "images" / "image_train"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
VAL_SIZE = 0.20

BEST_MODEL_PATH = OUTPUT_DIR / "best_model.keras"
FINAL_MODEL_PATH = OUTPUT_DIR / "final_model.keras"
HISTORY_PATH = OUTPUT_DIR / "history.json"
TRAIN_LOG_PATH = OUTPUT_DIR / "training_log.csv"
VAL_PREDICTIONS_PATH = OUTPUT_DIR / "val_predictions.csv"
CLASS_REPORT_PATH = OUTPUT_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.npy"
LABEL_MAP_PATH = OUTPUT_DIR / "label_mapping.json"
SUMMARY_JSON_PATH = OUTPUT_DIR / "summary.json"


# ============================================================
# 2. Utilities
# ============================================================

def log(msg):
    print(f"[INFO] {msg}", flush=True)


def save_json(obj, path):
    def convert(x):
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        if isinstance(x, tuple):
            return [convert(v) for v in x]
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2)

def history_to_dict(history_obj):
    return {k: [float(x) for x in v] for k, v in history_obj.history.items()}


def build_image_filename(imageid, productid):
    return f"image_{imageid}_product_{productid}.jpg"


# ============================================================
# 3. Load data
# ============================================================

log("Loading CSV files...")
X_df = pd.read_csv(X_TRAIN_CSV)
Y_df = pd.read_csv(Y_TRAIN_CSV)

if len(X_df) != len(Y_df):
    raise ValueError("X_train and Y_train row counts do not match.")

df = pd.concat(
    [
        X_df.reset_index(drop=True),
        Y_df.reset_index(drop=True)
    ],
    axis=1
)

log(f"Combined dataframe shape: {df.shape}")

# Build image paths
df["image_path"] = df.apply(
    lambda row: str(IMAGE_DIR / build_image_filename(row["imageid"], row["productid"])),
    axis=1
)

# Remove missing images
exists_mask = df["image_path"].map(os.path.exists)
missing = int((~exists_mask).sum())
if missing > 0:
    log(f"Removing {missing} rows with missing images")
df = df[exists_mask].reset_index(drop=True)

# Encode labels
label_values = sorted(df["prdtypecode"].unique())
label_to_index = {label: idx for idx, label in enumerate(label_values)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
df["label_idx"] = df["prdtypecode"].map(label_to_index)

NUM_CLASSES = len(label_values)

save_json(
    {
        "label_to_index": label_to_index,
        "index_to_label": index_to_label
    },
    LABEL_MAP_PATH
)

# Stratified split
train_df, val_df = train_test_split(
    df,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=df["label_idx"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

log(f"Train samples: {len(train_df)}")
log(f"Validation samples: {len(val_df)}")
log(f"Classes: {NUM_CLASSES}")


# ============================================================
# 4. tf.data pipeline
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(len(paths), seed=SEED)

    ds = ds.map(load_and_preprocess_image, num_parallel_calls=N_THREADS)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df["image_path"], train_df["label_idx"], True)
val_ds = make_dataset(val_df["image_path"], val_df["label_idx"], False)


# ============================================================
# 5. Build model
# ============================================================

inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

backbone = ResNet50(
    weights="imagenet",
    include_top=False,
    input_tensor=inputs
)
backbone.trainable = False

x = backbone.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(DROPOUT_RATE)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary(print_fn=log)


# ============================================================
# 6. Callbacks
# ============================================================

callback_list = [
    callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2
    ),
    callbacks.CSVLogger(str(TRAIN_LOG_PATH))
]


# ============================================================
# 7. Train
# ============================================================

log("Starting training...")
start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callback_list
)

training_time = time.time() - start_time
save_json(history_to_dict(history), HISTORY_PATH)


# ============================================================
# 8. Save final model
# ============================================================

model.save(FINAL_MODEL_PATH)


# ============================================================
# 9. Validation predictions
# ============================================================

best_model = tf.keras.models.load_model(BEST_MODEL_PATH)

val_probs = best_model.predict(val_ds)
val_pred_idx = np.argmax(val_probs, axis=1)
val_true_idx = val_df["label_idx"].values

val_accuracy = accuracy_score(val_true_idx, val_pred_idx)
log(f"Validation accuracy: {val_accuracy}")

val_pred_labels = [index_to_label[i] for i in val_pred_idx]
val_conf = val_probs.max(axis=1)

pred_df = val_df[["productid", "imageid", "prdtypecode"]].copy()
pred_df["pred"] = val_pred_labels
pred_df["confidence"] = val_conf
pred_df.to_csv(VAL_PREDICTIONS_PATH, index=False)

report = classification_report(val_true_idx, val_pred_idx)
with open(CLASS_REPORT_PATH, "w") as f:
    f.write(report)

cm = confusion_matrix(val_true_idx, val_pred_idx)
np.save(CONFUSION_MATRIX_PATH, cm)


# ============================================================
# 10. Summary
# ============================================================

summary = {
    "run_name": RUN_NAME,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "num_classes": NUM_CLASSES,
    "train_samples": len(train_df),
    "val_samples": len(val_df),
    "val_accuracy": float(val_accuracy),
    "training_time_sec": float(training_time)
}

save_json(summary, SUMMARY_JSON_PATH)

log("Training complete.")
