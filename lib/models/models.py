from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizedCosineHead(nn.Module):
    def __init__(self, in_features, num_classes, tau=0.15): # Nâng tau lên 0.15 để tránh overfitting nhãn đa số
        super().__init__()
        self.tau = tau
        # Trọng số đóng vai trò như các tâm lớp (class centers)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # 1. Chuẩn hóa L2 đặc trưng đầu vào
        x_norm = F.normalize(x, p=2, dim=1)
        # 2. Chuẩn hóa L2 trọng số tâm lớp
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # 3. Tính Cosine Similarity
        cosine_sim = F.linear(x_norm, w_norm)
        # 4. Scale bằng nhiệt độ tau
        return cosine_sim / self.tau
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
        self.head =NormalizedCosineHead(768, 4)
        print('encode head')

    def forward(self, x):
        x1 = self.Encoder.forward(x)
        x2 = self.head.forward(x1)
        x1_normalized = F.normalize(x1, p=2, dim=1)
        return x2, x1_normalized