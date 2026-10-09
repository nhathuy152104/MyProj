from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=512, tau=0.07):
        super().__init__()
        self.tau = tau

        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),     
            nn.Linear(hidden_dim, num_classes) # Khắc phục lỗi num_classes//2
        )
        
    def forward(self, x):
        z = self.projection(x)
        # Sử dụng temperature scaling (tau) để kiểm soát độ tự tin của phân phối xác suất
        return z
class MEDCLIPVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts, 768, bias = False)

    def forward(self, pixel_values, **kwargs):
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        img_embeds = self.model(pixel_values)
        return img_embeds




class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = MEDCLIPVisionModel()
        self.head = MLP(768, 4)
        print('encode head')

    def forward(self, x):
        x1 = self.Encoder.forward(x)
        x2 = self.head.forward(x1)
        x1_normalized = F.normalize(x1, p=2, dim=1)
        return x2, x1_normalized