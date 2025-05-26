# data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF

class AgeDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        # Map labels to integers
        self.label_map = {"YOUNG": 0, "MIDDLE": 1, "OLD": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['ID']
        label_str = row['Class']
        img_path = os.path.join(self.image_dir, img_name)
        # Load image
        image = Image.open(img_path).convert("RGB")
        # Convert to tensor
        image = TF.to_tensor(image)
        label = self.label_map[label_str]
        return image, label


def get_dataloaders(csv_file, image_dir, batch_size, val_split, seed):
    # Create full dataset
    full_dataset = AgeDataset(csv_file, image_dir)
    # Split indices for train/validation
    indices = list(range(len(full_dataset)))
    # Stratify by class label
    stratify_labels = full_dataset.df['Class']
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=stratify_labels
    )
    # Create subsets
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count()
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count()
    )
    return train_loader, val_loader
