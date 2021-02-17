import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from .models import ResNet
from .inception_resnet_v1 import InceptionResnetV1

class Resnet50(nn.Module):
    def __init__(self, pretrained=True, embedding_size=512):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)

class Backbone(nn.Module):
    def __init__(self, pretrained=True, embedding_size=512, device='cpu'):
        super(Backbone, self).__init__()
        # self.model = models.resnet50(pretrained=pretrained)
        # # self.model = InceptionResnetV1()
        # self.model.fc = nn.Sequential(
        #     nn.Linear(self.model.fc.in_features, embedding_size, bias=False),
        #     nn.BatchNorm1d(512)
        # )
        num_layers = 50
        mode = 'ir_se'
        drop_ratio = 0.6
        self.model = ResNet(num_layers, drop_ratio, mode)

    def forward(self, x):
        emb = self.model(x)
        return emb

class ShufflenetV2(nn.Module):
    def __init__(self, pretrained=True, embedding_size=512):
        super(ShufflenetV2, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_size)

    def forward(self, x):
        emb = self.model(x)
        return emb

class ArcMargin(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5, easy_margin=False, device='cpu'):
        super(ArcMargin, self).__init__()
        self.classnum = classnum
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)
        self.eps = 1e-6

    def forward(self, embbedings, label_onehot):
        cosine = F.linear(F.normalize(embbedings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        output = torch.where(label_onehot.bool(), phi, cosine)
        output *= self.s

        return output

class ArcFace(nn.Module):
    def __init__(self, classnum, backbone='shufflenetv2'):
        super(ArcFace, self).__init__()

        # if backbone == 'resnet50':
        #     self.backbone = Resnet50(pretrained=True, embedding_size=512)
        # elif backbone == 'shufflenetv2':
        #     self.backbone = ShufflenetV2(pretrained=True, embedding_size=512)
        self.backbone = Backbone()
        self.margin = ArcMargin(classnum=classnum)

    def forward(self, x, label):
        emb = self.backbone(x)
        output = self.margin(emb, label)
        return output