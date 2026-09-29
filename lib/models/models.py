from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, num_classes, tau=0.07):
        super().__init__()
        self.tau = tau
        
        # 1. Khối Projection (Nén và tạo phi tuyến tính)
        # Bắt buộc phải gom vào Sequential để code sạch sẽ
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),      # Dropout nên để trước hoặc sau ReLU
            nn.Linear(in_features, num_classes), # Bạn có thể đổi output thành hidden_dim (vd: 128)
        )
        
        # 2. Khối Phân loại Cosine (Thay thế hoàn toàn nn.Linear cuối)
        # Khởi tạo trọng số W như các "Mỏ neo ảo

    def forward(self, x):
        # Bước 1: Trích xuất đặc trưng qua MLP
        z = self.projection(x)
        
        return F.log_softmax(z, dim=1)
class MEDCLIPVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts, 512, bias = False)

    def forward(self, pixel_values, **kwargs):
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        img_embeds = self.model(pixel_values)
        return img_embeds




class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = MEDCLIPVisionModel()
        self.head = MLP(512, 4)
        print('encode head')

    def forward(self, x):
        x1 = self.Encoder.forward(x)
        x2 = self.head.forward(x1)
        x1_normalized = F.normalize(x1, p=2, dim=1)
        return x2, x1_normalized