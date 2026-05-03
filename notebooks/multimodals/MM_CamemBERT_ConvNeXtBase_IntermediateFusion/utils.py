# ============================================================
# utils.py
# Shared utilities: training loop, checkpointing, plotting,
# evaluation (accuracy, macro F1, weighted F1, confusion matrix)
# ============================================================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ------------------------------------------------------------
# Training / Validation loop
# ------------------------------------------------------------

def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple:
    """
    Run one full epoch (train or validation).

    Args:
        optimizer : Pass None for validation (no gradient updates).

    Returns:
        (epoch_loss, epoch_accuracy, epoch_macro_f1, epoch_weighted_f1)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds: list = []
    all_true:  list = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss   = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_true.extend(y.detach().cpu().numpy())

    epoch_loss        = total_loss / len(loader.dataset)
    epoch_acc         = accuracy_score(all_true, all_preds)
    epoch_macro_f1    = f1_score(all_true, all_preds, average="macro",    zero_division=0)
    epoch_weighted_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)

    return epoch_loss, epoch_acc, epoch_macro_f1, epoch_weighted_f1


# ------------------------------------------------------------
# Persistence
# ------------------------------------------------------------

def save_history_json(history: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_run_metadata(
    path: Path,
    model_name: str,
    best_f1: float,
    best_epoch: int,
    duration_min: float,
) -> None:
    meta = {
        "model_name":    model_name,
        "best_macro_f1": round(best_f1, 6),
        "best_epoch":    best_epoch,
        "duration_min":  round(duration_min, 2),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {path}")


# ------------------------------------------------------------
# Plotting – training curves
# ------------------------------------------------------------

def plot_and_save_history(
    history: list,
    fig_dir: Path,
    best_epoch: int,
    best_f1: float,
) -> None:
    """2×2 grid: Loss / Accuracy / Macro-F1 / Weighted-F1."""
    if not history:
        print("No history to plot.")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Training Curves  (Best Epoch: {best_epoch} | Best Macro-F1: {best_f1:.4f})",
            fontsize=13,
        )

        plots = [
            ("train_loss",        "val_loss",        "Loss",        axes[0, 0]),
            ("train_acc",         "val_acc",         "Accuracy",    axes[0, 1]),
            ("train_macro_f1",    "val_macro_f1",    "Macro F1",    axes[1, 0]),
            ("train_weighted_f1", "val_weighted_f1", "Weighted F1", axes[1, 1]),
        ]

        for train_col, val_col, title, ax in plots:
            if train_col in df.columns:
                ax.plot(df["epoch"], df[train_col], "-o", label="Train", linewidth=2)
            if val_col in df.columns:
                ax.plot(df["epoch"], df[val_col],   "-o", label="Val",   linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(fig_dir / "training_curves.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved to: {fig_dir / 'training_curves.png'}")

    except Exception as e:
        print(f"Error saving training curves: {e}")


# ------------------------------------------------------------
# Plotting – confusion matrix
# ------------------------------------------------------------

def plot_confusion_matrix(
    all_true: list,
    all_preds: list,
    target_names: list,
    fig_dir: Path,
    model_name: str,
    macro_f1: float,
    normalise: bool = True,
) -> None:
    """Compute and save a confusion matrix heatmap."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(all_true, all_preds)
    if normalise:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt, title_tag = ".2f", "Normalised"
    else:
        fmt, title_tag = "d", "Absolute"

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        annot=False,
        fmt=fmt,
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title(
        f"{title_tag} Confusion Matrix – {model_name}  (Macro-F1: {macro_f1:.4f})",
        fontsize=14,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {fig_dir / 'confusion_matrix.png'}")


# ------------------------------------------------------------
# Full evaluation  (used by evaluate.py)
# ------------------------------------------------------------

def evaluate_model(
    model,
    loader,
    device,
    target_names: list,
    fig_dir: Path,
    model_name: str,
) -> dict:
    """
    Run inference, print full classification report, save confusion matrix.

    Returns:
        {"accuracy": float, "macro_f1": float, "weighted_f1": float}
    """
    model.eval()
    all_preds: list = []
    all_true:  list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds  = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.numpy())

    acc         = accuracy_score(all_true, all_preds)
    macro_f1    = f1_score(all_true, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"Evaluation Results – {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(all_true, all_preds, target_names=target_names, zero_division=0))

    plot_confusion_matrix(all_true, all_preds, target_names, fig_dir, model_name, macro_f1)

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
