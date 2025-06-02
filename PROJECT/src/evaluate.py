# evaluate.py

import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from config import MODEL_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_SEED, VAL_SPLIT
from data_loader import AgeRegressionDataset, get_regression_transform
from models.resnet50 import build_resnet50_regression


def evaluate(model_path=None):
    # ─── Model Path ────────────────────────────────────────────────────
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "best_resnet50.pth")
    print(f"\n[INFO] Using model path: {model_path}")

    # ─── Device Setup ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running evaluation on device: {device}")

    # ─── Dataset & Dataloader ───────────────────────────────────────────
    transform = get_regression_transform()
    full_dataset = AgeRegressionDataset(
        image_dir=os.path.join(os.path.dirname(__file__), "..", "data", "processed"),
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

    print(f"[INFO] Total samples in dataset: {total_samples}")
    print(f"[INFO] Split → Train: {train_size}, Validation: {val_size}\n")

    # ─── Model Loading ──────────────────────────────────────────────────
    model = build_resnet50_regression(pretrained=False).to(device)
    print("[INFO] Loading model weights...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("[INFO] Model loaded successfully.\n")

    # ─── Evaluation Loop ────────────────────────────────────────────────
    all_preds = []
    all_targets = []
    batch_errors = []

    print("[INFO] Starting evaluation on validation set...\n")
    with torch.no_grad():
        for batch_idx, (imgs, ages) in enumerate(tqdm(val_loader, desc="Evaluating", unit="batch")):
            # Move to device
            imgs = imgs.to(device)
            ages = ages.to(device).unsqueeze(1)  # shape: (batch_size, 1)

            # Forward pass
            outputs = model(imgs)
            preds = outputs.cpu().squeeze().tolist()
            targets = ages.cpu().squeeze().tolist()

            # Collect predictions and targets
            all_preds.extend(preds)
            all_targets.extend(targets)

            # Compute and log batch-level MAE
            batch_mae = mean_absolute_error(targets, preds)
            batch_mse = mean_squared_error(targets, preds)
            batch_rmse = batch_mse ** 0.5
            batch_errors.append(batch_mae)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                print(
                    f"[BATCH {batch_idx + 1}/{len(val_loader)}] "
                    f"Batch MAE: {batch_mae:.3f}, Batch RMSE: {batch_rmse:.3f}"
                )

    # ─── Aggregate Metrics ───────────────────────────────────────────────
    overall_mae = mean_absolute_error(all_targets, all_preds)
    overall_mse = mean_squared_error(all_targets, all_preds)
    overall_rmse = overall_mse ** 0.5

    print("\n[RESULT] ===================== Validation Metrics =====================")
    print(f" - Samples evaluated: {len(all_targets)}")
    print(f" - Mean Absolute Error (MAE): {overall_mae:.3f}")
    print(f" - Root Mean Squared Error (RMSE): {overall_rmse:.3f}")
    print("======================================================================\n")

    # ─── Detailed Logging: Sample Predictions ────────────────────────────
    print("[INFO] Example predictions vs. ground truth (first 10):")
    for i in range(min(10, len(all_targets))):
        print(f"   Sample {i+1}: Predicted Age = {all_preds[i]:.1f}, True Age = {all_targets[i]:.1f}")


if __name__ == "__main__":
    evaluate()
