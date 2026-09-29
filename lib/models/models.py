from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
import torch

class MLP(nn.Module):
    def __init__(self, in_features, num_classes, tau=0.07):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.tau = tau

    def forward(self, x):
        # 1. Chuẩn hóa x và trọng số W lên mặt cầu
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 2. Tính Cosine thay vì tích vô hướng
        logits = F.linear(x_norm, w_norm) / self.tau
        return logits
class MEDCLIPVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50()
        num_fts = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fts, 512, bias = False)

    def forward(self, pixel_values, **kwargs):
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        img_embeds = self.model(pixel_values)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        return img_embeds




class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = MEDCLIPVisionModel()
        checkpoint_path = '/kaggle/input/models/huynhat15/gogo/pytorch/default/1/client_0_round_0.pth.tar'
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # 3. Nạp Dictionary trọng số vào mô hình
        self.Encoder.load_state_dict(state_dict)
        print('encode head')

    def forward(self, x):
        x1 = self.Encoder.forward(x)
        x2 = self.head.forward(x1)
        x1_normalized = F.normalize(x1, p=2, dim=1)
        return x2, x1_normalized