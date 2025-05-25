
import torch
from sklearn.metrics import classification_report, confusion_matrix
from config import *
from data_loader import get_dataloaders
from models.resnet18 import build_resnet18


def evaluate(model_path):
    _, val_loader = get_dataloaders(
        CSV_FILE, RAW_DIR, BATCH_SIZE, VAL_SPLIT, RANDOM_SEED
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=["YOUNG", "MIDDLE", "OLD"]))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
