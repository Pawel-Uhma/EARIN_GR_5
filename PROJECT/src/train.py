import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import NUM_EPOCHS, LEARNING_RATE, MODEL_DIR
from data_loader import get_dataloaders
from models.resnet import build_resnet_regression
from utils import set_seed, plot_train_val_loss


def train():
    # set the seed so results are reproducible every time
    set_seed()

    # grab train and val dataloaders
    train_loader, val_loader = get_dataloaders()

    # use gpu if available, else fallback to cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load a resnet18 model and move to correct device
    model = build_resnet_regression(pretrained=True).to(device)

    # standard regression loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')  # to keep track of the best model
    train_losses, val_losses = [], []  # for plotting later

    # training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_train = 0.0

        # train on all batches
        for imgs, ages in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Training"):
            imgs, ages = imgs.to(device), ages.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * imgs.size(0)  # track total loss

        epoch_train_loss = running_train / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # validation phase
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, ages in val_loader:
                imgs, ages = imgs.to(device), ages.to(device).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, ages)
                running_val += loss.item() * imgs.size(0)

        epoch_val_loss = running_val / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # log the epoch result
        print(f"Epoch {epoch}: Train Loss={epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}")

        # save model if it's the best so far
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_resnet18.pth")

    print(f"Training complete! Best Val Loss: {best_loss:.4f}")

    # plot training + validation losses across epochs
    plot_train_val_loss(train_losses, val_losses)


if __name__ == '__main__':
    train()
