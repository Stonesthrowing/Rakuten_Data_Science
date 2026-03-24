# ============================================================
# train_cnn_256.py
#
# Rakuten multimodal project
# Image-only CNN baseline on 256x256 images
#
# Uses:
# - data/raw/X_train.csv
# - data/raw/Y_train.csv
# - data/raw/images/image_train/
#
# Saves:
# - data/raw/cnn_256_best.keras
# - data/raw/cnn_256_final.keras
# - data/raw/cnn_256_history.csv
# - data/raw/cnn_256_summary.json
# - data/raw/cnn_256_val_predictions.csv
# - data/raw/cnn_256_class_indices.json
# ============================================================

import os

# Match this to Slurm --cpus-per-task
N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(N_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# ------------------------------------------------------------
# TensorFlow threading
# ------------------------------------------------------------
tf.config.threading.set_intra_op_parallelism_threads(N_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(2)

print("=" * 70)
print("TensorFlow version:", tf.__version__)
print("Intra-op threads  :", tf.config.threading.get_intra_op_parallelism_threads())
print("Inter-op threads  :", tf.config.threading.get_inter_op_parallelism_threads())
print("=" * 70)

# ------------------------------------------------------------
# Paths
# If RAKUTEN_DATA_DIR is set in the Slurm job, use that.
# Otherwise fall back to project-root/data/raw.
# ------------------------------------------------------------
raw_dir_env = os.environ.get("RAKUTEN_DATA_DIR")

if raw_dir_env is not None:
    RAW_DIR = Path(raw_dir_env)
    BASE_DIR = RAW_DIR.parent.parent
else:
    BASE_DIR = Path(__file__).resolve().parent
    RAW_DIR = BASE_DIR / "data" / "raw"

TRAIN_IMAGES_DIR = RAW_DIR / "images" / "image_train"

X_TRAIN_PATH = RAW_DIR / "X_train.csv"
Y_TRAIN_PATH = RAW_DIR / "Y_train.csv"
OUT_DIR = RAW_DIR.parent / "outputs"

MODEL_BEST_PATH = OUT_DIR / "cnn_256_best.keras"
MODEL_FINAL_PATH = OUT_DIR / "cnn_256_final.keras"
HISTORY_PATH = OUT_DIR / "cnn_256_history.csv"
SUMMARY_PATH = OUT_DIR / "cnn_256_summary.json"
PREDS_PATH = OUT_DIR / "cnn_256_val_predictions.csv"
CLASS_INDICES_PATH = OUT_DIR / "cnn_256_class_indices.json"

print("BASE_DIR         :", BASE_DIR)
print("RAW_DIR          :", RAW_DIR)
print("TRAIN_IMAGES_DIR :", TRAIN_IMAGES_DIR)
print("X_TRAIN_PATH     :", X_TRAIN_PATH)
print("Y_TRAIN_PATH     :", Y_TRAIN_PATH)
print("=" * 70)

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
print("\nLoading X_train and Y_train...")
X_train = pd.read_csv(X_TRAIN_PATH)
Y_train = pd.read_csv(Y_TRAIN_PATH)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_train columns:", X_train.columns.tolist())
print("Y_train columns:", Y_train.columns.tolist())

# ------------------------------------------------------------
# Merge features and labels
# ------------------------------------------------------------
df = pd.concat(
    [
        X_train.reset_index(drop=True),
        Y_train.reset_index(drop=True)
    ],
    axis=1
)

print("\nMerged dataframe shape:", df.shape)

# ------------------------------------------------------------
# Build image paths
# ------------------------------------------------------------
df["image_path"] = df.apply(
    lambda row: str(TRAIN_IMAGES_DIR / f"image_{row['imageid']}_product_{row['productid']}.jpg"),
    axis=1
)

print("\nExample image paths:")
print(df["image_path"].head())

# ------------------------------------------------------------
# Fixed stratified split
# ------------------------------------------------------------
print("\nCreating fixed stratified split...")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["prdtypecode"],
    random_state=42
)

print("Train size      :", len(train_df))
print("Validation size :", len(val_df))

train_df = train_df.copy()
val_df = val_df.copy()
train_df["prdtypecode"] = train_df["prdtypecode"].astype(str)
val_df["prdtypecode"] = val_df["prdtypecode"].astype(str)

# ------------------------------------------------------------
# tf.data input pipeline
# ------------------------------------------------------------
IMG_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 15
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Class mapping
class_names = sorted(train_df["prdtypecode"].unique())
class_to_idx = {label: idx for idx, label in enumerate(class_names)}
idx_to_class = {idx: label for label, idx in class_to_idx.items()}
NUM_CLASSES = len(class_names)

with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, indent=2)

print("Number of classes:", NUM_CLASSES)

# Integer labels
train_labels_int = train_df["prdtypecode"].map(class_to_idx).astype(np.int32).values
val_labels_int = val_df["prdtypecode"].map(class_to_idx).astype(np.int32).values

# Paths
train_paths = train_df["image_path"].values
val_paths = val_df["image_path"].values

# Datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels_int))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels_int))

train_ds = train_ds.shuffle(buffer_size=len(train_df), seed=SEED)

train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.map(
    lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)),
    num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.map(
    lambda x, y: (x, tf.one_hot(y, depth=NUM_CLASSES)),
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("tf.data pipelines created.")
print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
print("\nBuilding CNN model...")

model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------------
# Timing callback
# ------------------------------------------------------------
class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"\nEpoch {epoch + 1} time: {epoch_time:.2f} sec ({epoch_time / 60:.2f} min)")

time_callback = TimeHistory()

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=str(MODEL_BEST_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    time_callback
]

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
print("\nStarting training...")

train_start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

total_training_time = time.time() - train_start_time

print("\nTraining finished.")
print(f"Total training time: {total_training_time:.2f} sec ({total_training_time / 60:.2f} min)")

# ------------------------------------------------------------
# Save final model
# ------------------------------------------------------------
model.save(MODEL_FINAL_PATH)
print("Saved final model to:", MODEL_FINAL_PATH)

# ------------------------------------------------------------
# Save history
# ------------------------------------------------------------
history_df = pd.DataFrame(history.history)
history_df["epoch_time_sec"] = time_callback.epoch_times
history_df["epoch_time_min"] = history_df["epoch_time_sec"] / 60.0
history_df.to_csv(HISTORY_PATH, index=False)

best_val_accuracy = float(history_df["val_accuracy"].max())
best_val_loss = float(history_df["val_loss"].min())
avg_epoch_time_sec = float(np.mean(time_callback.epoch_times))
completed_epochs = int(len(history_df))

summary = {
    "experiment": "cnn_baseline_256",
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "epochs_completed": completed_epochs,
    "num_classes": NUM_CLASSES,
    "train_size": int(len(train_df)),
    "val_size": int(len(val_df)),
    "best_val_accuracy": best_val_accuracy,
    "best_val_loss": best_val_loss,
    "avg_epoch_time_sec": avg_epoch_time_sec,
    "avg_epoch_time_min": avg_epoch_time_sec / 60.0,
    "total_training_time_sec": float(total_training_time),
    "total_training_time_min": float(total_training_time / 60.0),
    "model_best_path": str(MODEL_BEST_PATH),
    "model_final_path": str(MODEL_FINAL_PATH),
    "history_path": str(HISTORY_PATH),
    "predictions_path": str(PREDS_PATH),
    "class_indices_path": str(CLASS_INDICES_PATH)
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved history to:", HISTORY_PATH)
print("Saved summary to:", SUMMARY_PATH)
print("Saved class mapping to:", CLASS_INDICES_PATH)

# ------------------------------------------------------------
# Save validation predictions for later confusion matrix
# ------------------------------------------------------------
print("\nGenerating validation predictions...")

pred_probs = model.predict(val_ds, verbose=1)
pred_idx = np.argmax(pred_probs, axis=1)
pred_labels = [idx_to_class[i] for i in pred_idx]
true_labels = val_df.reset_index(drop=True)["prdtypecode"].tolist()

val_results_df = val_df.reset_index(drop=True).copy()
val_results_df["true_label"] = true_labels
val_results_df["pred_label"] = pred_labels
val_results_df["pred_correct"] = (
    val_results_df["true_label"] == val_results_df["pred_label"]
).astype(int)

val_results_df.to_csv(PREDS_PATH, index=False)
print("Saved validation predictions to:", PREDS_PATH)

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Best validation accuracy : {best_val_accuracy:.6f}")
print(f"Best validation loss     : {best_val_loss:.6f}")
print(f"Epochs completed         : {completed_epochs}")
print(f"Average epoch time       : {avg_epoch_time_sec:.2f} sec ({avg_epoch_time_sec / 60:.2f} min)")
print(f"Total training time      : {total_training_time:.2f} sec ({total_training_time / 60:.2f} min)")
print("\nArtifacts saved in data/raw:")
print("-", MODEL_BEST_PATH.name)
print("-", MODEL_FINAL_PATH.name)
print("-", HISTORY_PATH.name)
print("-", SUMMARY_PATH.name)
print("-", PREDS_PATH.name)
print("-", CLASS_INDICES_PATH.name)
print("=" * 70)
