import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import MODEL_DIR
from data_loader import get_dataloaders
from models.resnet18 import build_resnet18_regression
from utils import get_regression_transform


def evaluate(model_path=None):
    if model_path is None:
        model_path = f"{MODEL_DIR}/best_resnet50.pth"
    train_loader, val_loader = get_dataloaders(transform=get_regression_transform())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_resnet18_regression(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, ages in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs).cpu().squeeze().tolist()
            y_pred.extend(outputs)
            y_true.extend(ages.tolist())

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"Evaluation Results - MAE: {mae:.2f}, RMSE: {mse**0.5:.2f}")

if __name__ == '__main__':
    evaluate()