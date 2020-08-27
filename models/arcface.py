import math

import torch
from torch.nn import functional as F
from torch import nn
from torchvision import models


class ArcFace(nn.Module):
    def __init__(self, pretrained=True, embedding_size=2, classnum=5, s=64., m=0.5, easy_margin=False,
                 device='cpu', training=True):
        super(ArcFace, self).__init__()
        self.training = training
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)

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

    def forward(self, x, label):
        emb = self.backbone(x)

        if not self.training:
            return emb

        else:
            cosine = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + self.eps, 1 - self.eps)
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

            one_hot = torch.zeros(cosine.size(), device=self.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            output = torch.where(one_hot.bool(), phi, cosine)
            output *= self.s

            return output
