from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class MEDCLIPVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resn3et50(pretrained=True)
        num_fts = self.model.in_feature
        self.model.fc = nn.Linear(num_fts, 512, bias = False)

    def forward(self, pixel_values, **kwargs):
        if pixel_values.shape[1] == 1: pixel_values = pixel_values.repeat((1,3,1,1))
        img_embeds = self.model(pixel_values)
        return img_embeds




class ClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = MEDCLIPVisionModel()
        self.head = MLP(512, 512, 4)

    def forward(self, x):
        x1 = self.Encoder.forward(x)
        x2 = self.head.forward(x1)

        return x2, x1