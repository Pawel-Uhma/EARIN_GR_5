
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from sklearn.model_selection import train_test_split
from preprocessing import get_transforms

class AgeDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        # Map labels to integers
        self.label_map = {"YOUNG": 0, "MIDDLE": 1, "OLD": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label_str = self.df.iloc[idx, 1]
        img_path = f"{self.image_dir}/{img_name}"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_map[label_str]
        return image, label


def get_dataloaders(csv_file, image_dir, batch_size, val_split, seed):
    # Full dataset with transforms for training
    full_dataset = AgeDataset(csv_file, image_dir, transform=get_transforms('train'))
    # Split indices
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=full_dataset.df['Class']
    )
    # Datasets for train/val
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(
        AgeDataset(csv_file, image_dir, transform=get_transforms('val')),
        val_idx
    )
    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader