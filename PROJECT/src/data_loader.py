import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from config import IMAGE_DIR, BATCH_SIZE, VAL_SPLIT, RANDOM_SEED, IMG_SIZE
import pandas as pd

class AgeRegressionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        for fname in os.listdir(image_dir):
            if not fname.lower().endswith('.jpg'):
                continue

            parts = fname.split('_', 3)
            if len(parts) < 4:
                continue

            try:
                age = float(parts[0])
                gender = int(parts[1])   
                race   = int(parts[2])   
            except ValueError:
                continue

            path = os.path.join(image_dir, fname)
            self.samples.append((path, age))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, age = self.samples[idx]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(age, dtype=torch.float32)


def get_regression_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders():
    full_dataset = AgeRegressionDataset(
        image_dir=IMAGE_DIR,
        transform=get_regression_transform()
    )

    total_len = len(full_dataset)
    val_len   = int(total_len * VAL_SPLIT)
    train_len = total_len - val_len

    train_ds, val_ds = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count()
    )

    return train_loader, val_loader
