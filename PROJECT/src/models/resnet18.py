import torch.nn as nn
import torchvision.models as models

def build_resnet18_regression(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    # modify final layer for single-output regression
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    return model