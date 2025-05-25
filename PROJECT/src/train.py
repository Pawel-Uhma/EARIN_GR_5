import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import *
from data_loader import get_dataloaders
from models.resnet18 import build_resnet18
from utils import set_seed


def train():
    set_seed(RANDOM_SEED)
    train_loader, val_loader = get_dataloaders(
        CSV_FILE, RAW_DIR, BATCH_SIZE, VAL_SPLIT, RANDOM_SEED
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(MODEL_DIR, "best_resnet18.pth")
            torch.save(model.state_dict(), save_path)

    print("Training complete. Best Val Acc: {:.4f}".format(best_acc))

if __name__ == "__main__":
    train()
