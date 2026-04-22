# ============================================================
# MODEL G+ - ConvNeXt-Base (PRO FINETUNING)
# Optimized for RTX 5070 Ti & Windows Multiprocessing
# Logit Export for Late Fusion included
# ============================================================

import gc
import os
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.amp import GradScaler, autocast 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ------------------------------------------------------------
# 1. Dataset & Helpers
# ------------------------------------------------------------
class RakutenDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path_local"]
        label = int(self.df.loc[idx, "label_id"])
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform: image = self.transform(image)
        except Exception:
            image = torch.zeros((3, 224, 224))
        return image, label, idx

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    running_loss, all_preds, all_true = 0.0, [], []

    for images, labels, _ in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if is_train: optimizer.zero_grad()
        
        with autocast(device_type='cuda', enabled=True): 
            logits = model(images)
            loss = criterion(logits, labels)
        
        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_true.extend(labels.cpu().numpy())

    return running_loss/len(loader.dataset), f1_score(all_true, all_preds, average="macro")

@torch.no_grad()
def export_logits(model, loader, device, output_path, name):
    model.eval()
    all_logits = []
    print(f"🚀 Exporting Logits: {name}...")
    for images, _, _ in loader:
        images = images.to(device)
        with autocast(device_type='cuda'):
            logits = model(images)
        all_logits.append(logits.cpu().numpy())
    
    logits_array = np.vstack(all_logits)
    np.save(output_path / f"{name}_logits_base.npy", logits_array)

# ------------------------------------------------------------
# 2. Main Execution Block
# ------------------------------------------------------------
def main():
    # Hardware & Seed Setup
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False # Für feste Inputgrößen schneller
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pfade (Synchronisiert auf deine Struktur)
    PROJECT_DIR = Path(r"C:\Users\felix\Documents\DS_MLE_MasterSystem\06_PROJECTS\Project_01_Rakuten_Multimodal")
    RAW_IMG_DIR = PROJECT_DIR / "data" / "raw" / "images" / "image_train"
    SPLIT_DIR = PROJECT_DIR / "outputs" / "image_modeling"
    
    LOCAL_RUN_NAME = "convnext_base_final_v1"
    LOCAL_OUTPUT_ROOT = PROJECT_DIR / "outputs" / LOCAL_RUN_NAME
    LOCAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Hyperparameter
    IMAGE_SIZE = 224
    BATCH_SIZE = 16 # VRAM Limit Base Modell
    NUM_WORKERS = 2 # Dein Thermal-Limit
    MAX_EPOCHS = 20
    LR_HEAD = 5e-5
    LR_BACKBONE = 5e-6
    WEIGHT_DECAY = 0.05

    # Data Loading
    train_df = pd.read_csv(SPLIT_DIR / "train_split.csv")
    val_df = pd.read_csv(SPLIT_DIR / "val_split.csv")
    # Kein Test-DF, da keine Labels vorhanden

    with open(SPLIT_DIR / "label2id.json", "r") as f:
        label2id = json.load(f)

    def get_path(r): return str(RAW_IMG_DIR / f"image_{r['imageid']}_product_{r['productid']}.jpg")
    for df in [train_df, val_df]:
        df["image_path_local"] = df.apply(get_path, axis=1)
        df["label_id"] = df["prdtypecode"].astype(str).map({str(k): v for k,v in label2id.items()})

    # Transforms (Synchronisiert!)
    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_trans = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(RakutenDataset(train_df, train_trans), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(RakutenDataset(val_df, val_trans), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    
    # Model Setup
    model = models.convnext_base(weights='IMAGENET1K_V1')
    n_inputs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(n_inputs, len(label2id))
    model = model.to(device)

    # Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda')

    # Training Loop
    best_f1 = 0
    history = []

    print(f"Starting Base Training on {device}...")
    for epoch in range(1, MAX_EPOCHS + 1):
        t_loss, t_f1 = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        v_loss, v_f1 = run_epoch(model, val_loader, criterion, device)

        history.append({"epoch": epoch, "t_loss": t_loss, "t_f1": t_f1, "v_loss": v_loss, "v_f1": v_f1})
        print(f"Epoch {epoch:02d} | t_loss: {t_loss:.4f} | v_f1: {v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), LOCAL_OUTPUT_ROOT / "best_model_base.pt")
            print(" ⭐ New Best Model!")

    print(f"🎉 Training Finished. Best Val F1: {best_f1:.4f}")
    # ============================================================
    # 6. FINALE EVALUATION, PLOTS & LOGITS
    # ============================================================
    print("\n🏁 Training beendet. Starte finale Evaluation auf dem Val-Set...")
    model.load_state_dict(torch.load(LOCAL_OUTPUT_ROOT / "best_model_base.pt"))
    # Metriken für Test-Set berechnen
   
    # Für Confusion Matrix und Report müssen wir einmal "manuell" durch
    all_preds, all_true = [], []
    model.eval()
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            with autocast(device_type='cuda'):
                outputs = model(images)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_true.extend(labels.numpy())

    # 1. Classification Report speichern
    report = classification_report(all_true, all_preds, target_names=[str(i) for i in range(27)])
    with open(LOCAL_OUTPUT_ROOT / "val_classification_report.txt", "w") as f:
        f.write(report)
    print("\nValidation Classification Report:\n", report)

    # 2. Confusion Matrix Plot
    plt.figure(figsize=(16, 12))
    cm = confusion_matrix(all_true, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=False, cmap='Blues')
    plt.title(f"{LOCAL_RUN_NAME} - Normalized Val Confusion Matrix (Best F1: {best_f1:.4f})")
    plt.savefig(LOCAL_OUTPUT_ROOT / "val_confusion_matrix.png", dpi=300)
    plt.close()

    # 3. Learning Curves Plot
    history_df = pd.DataFrame(history)
    history_df.to_csv(LOCAL_OUTPUT_ROOT / "history.csv", index=False)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(history_df['epoch'], history_df['t_loss'], 'g-', label='Train Loss')
    ax1.plot(history_df['epoch'], history_df['v_loss'], 'b-', label='Val Loss')
    ax2.plot(history_df['epoch'], history_df['v_f1'], 'r--', label='Val Macro F1')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='g')
    ax2.set_ylabel('F1 Score', color='r')
    plt.title(f"Training History - {LOCAL_RUN_NAME}")
    plt.grid(True)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.savefig(LOCAL_OUTPUT_ROOT / "learning_curves.png", dpi=300)
    plt.close()

    # 4. Logit Export für Late Fusion (Val & Test)
    export_logits(model, val_loader, device, LOCAL_OUTPUT_ROOT, "val")
    
    print(f"✅ Alles erledigt! Logits für Fusion gespeichert in: {LOCAL_OUTPUT_ROOT}")
    
    

if __name__ == '__main__':
    main()