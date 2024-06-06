import torch
import torch.nn as nn
from torchvision.models import densenet121 , DenseNet121_Weights
from config import DEVICE

model = densenet121(DenseNet121_Weights.IMAGENET1K_V1).to(device=DEVICE)


model.classifier = nn.Linear(in_features=1024 , out_features=14)

# unfreezing all parameters
for param in model.parameters():
    param.requires_grad = True

# freezing only 1st dense layer 
for param in model.features[4].parameters():
    param.requires_grad = False


model.to(device=DEVICE)





