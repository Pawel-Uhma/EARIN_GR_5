# evaluate.py

import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from config import MODEL_DIR, IMAGE_DIR, BATCH_SIZE, RANDOM_SEED, VAL_SPLIT
from data_loader import AgeRegressionDataset, get_regression_transform
from models.resnet18 import build_resnet18_regression
from utils import (
    plot_predictions_vs_truth,
    plot_error_distribution,
    plot_residuals_vs_true,
    plot_true_age_distribution,
    plot_error_by_age_bin
)


def evaluate(model_path=None):
    # ─── Model Path ────────────────────────────────────────────────────
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "best_resnet18.pth")
    print(f"\n[INFO] Using model path: {model_path}")

    # ─── Device Setup ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running evaluation on device: {device}")

    # ─── Dataset & Dataloader ───────────────────────────────────────────
    transform = get_regression_transform()
    full_dataset = AgeRegressionDataset(
        image_dir=IMAGE_DIR,
        transform=transform
    )
    total_samples = len(full_dataset)
    val_size = int(total_samples * VAL_SPLIT)
    train_size = total_samples - val_size

    # Ensure reproducibility
    torch.manual_seed(RANDOM_SEED)
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    print(f"[INFO] Total samples: {total_samples} (Train: {train_size}, Val: {val_size})\n")

    # ─── Model Loading ──────────────────────────────────────────────────
    model = build_resnet18_regression(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("[INFO] Model loaded successfully.\n")

    # ─── Evaluation Loop ────────────────────────────────────────────────
    all_preds = []
    all_targets = []

    print("[INFO] Starting evaluation...\n")
    with torch.no_grad():
        for imgs, ages in tqdm(val_loader, desc="Evaluating", unit="batch"):
            imgs = imgs.to(device)
            ages = ages.to(device).unsqueeze(1)
            outputs = model(imgs).cpu().squeeze().tolist()
            targets = ages.cpu().squeeze().tolist()

            all_preds.extend(outputs)
            all_targets.extend(targets)

    # ─── Compute Metrics ────────────────────────────────────────────────
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = mse ** 0.5

    print("\n[RESULT] ===================== Validation Metrics =====================")
    print(f" MAE:  {mae:.3f}")
    print(f" RMSE: {rmse:.3f}")
    print("======================================================================\n")

    # ─── Generate & Save Plots ──────────────────────────────────────────
    plot_predictions_vs_truth(all_targets, all_preds)
    plot_error_distribution(all_targets, all_preds)
    plot_residuals_vs_true(all_targets, all_preds)
    plot_true_age_distribution(all_targets)
    plot_error_by_age_bin(all_targets, all_preds)

    # ─── Log Example Predictions ────────────────────────────────────────
    print("[INFO] Sample predictions vs. ground truth (first 10):")
    for i in range(min(10, len(all_targets))):
        print(f"  #{i+1}: Pred={all_preds[i]:.1f}, True={all_targets[i]:.1f}")


if __name__ == "__main__":
    evaluate()
