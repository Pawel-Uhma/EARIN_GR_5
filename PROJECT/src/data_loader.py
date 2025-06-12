import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from config import IMAGE_DIR, BATCH_SIZE, VAL_SPLIT, RANDOM_SEED, IMG_SIZE

# ------------------------------------------------------------------
# custom dataset
# ------------------------------------------------------------------
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
                age = float(parts[0])          # we only keep age
                gender = int(parts[1])         # parsed but not used (kept for future work)
                race   = int(parts[2])
            except ValueError:
                continue

            self.samples.append((os.path.join(image_dir, fname), age))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, age = self.samples[idx]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(age, dtype=torch.float32)

# ------------------------------------------------------------------
# transforms
# ------------------------------------------------------------------
def get_train_transform():
    """data-augmentation for training (no rotation)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            IMG_SIZE, scale=(0.9, 1.0), ratio=(0.9, 1.1)
        ),                                # slight random crop / zoom
        transforms.RandomHorizontalFlip(),# flip left↔right 50% of the time
        transforms.ColorJitter(           # small random brightness/contrast/etc.
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def get_val_transform():
    """simple resize for validation / test."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

# ------------------------------------------------------------------
# dataloaders
# ------------------------------------------------------------------
def get_dataloaders():
    full_dataset = AgeRegressionDataset(
        IMAGE_DIR,
        transform=None  # temp placeholder
    )

    total_len = len(full_dataset)
    val_len   = int(total_len * VAL_SPLIT)
    train_len = total_len - val_len

    # split once for reproducibility
    torch.manual_seed(RANDOM_SEED)
    train_idx, val_idx = torch.utils.data.random_split(
        range(total_len),
        [train_len, val_len],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # wrap indices back into dataset objects with their own transforms
    train_ds = torch.utils.data.Subset(
        AgeRegressionDataset(IMAGE_DIR, transform=get_train_transform()), train_idx.indices
    )
    val_ds = torch.utils.data.Subset(
        AgeRegressionDataset(IMAGE_DIR, transform=get_val_transform()), val_idx.indices
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=os.cpu_count()
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=os.cpu_count()
    )

    return train_loader, val_loader
