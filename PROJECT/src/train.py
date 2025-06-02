import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import NUM_EPOCHS, LEARNING_RATE, MODEL_DIR
from data_loader import get_dataloaders
from models.resnet18 import build_resnet18_regression
from utils import set_seed, get_regression_transform


def train():
    set_seed()
    train_loader, val_loader = get_dataloaders(transform=get_regression_transform())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_resnet18_regression(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, ages in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}" ):
            imgs, ages = imgs.to(device), ages.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, ages in val_loader:
                imgs, ages = imgs.to(device), ages.to(device).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, ages)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        # save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_resnet50.pth")

    print(f"Training complete! Best Val Loss: {best_loss:.4f}")

if __name__ == '__main__':
    train()