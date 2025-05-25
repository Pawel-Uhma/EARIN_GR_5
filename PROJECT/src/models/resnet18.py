import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES

def build_resnet18(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    # Replace final layer for classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model
