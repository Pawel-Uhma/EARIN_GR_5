import torch.nn as nn
import torchvision.models as models
from config import RESNET

def build_resnet_regression(pretrained=True):
    # load resnet
    if RESNET == 18:
        model = models.resnet18(pretrained=pretrained)
    else: #50
        model = models.resnet50(pretrained=pretrained)
    
    # the original last layer is for classification (usually 1000 classes),
    # but we only want to predict a single continuous value (age), so we replace it
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # this makes it a regression model
    
    return model
