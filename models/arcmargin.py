import math

import torch
from torch import functional as F
from torch import nn


class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5, easy_margin=False, device='cpu'):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)
        self.eps = 1e-6

    def forward(self, embbedings, label):
        cosine = F.linear(F.normalize(embbedings), F.normalize(self.weight)).clamp(-1 + self.eps, 1 - self.eps)
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
