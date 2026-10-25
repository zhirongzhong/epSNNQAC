"""Convolutional feature extraction and EP moment propagation."""

import math

import torch
from torch import nn


class MS_CAM_1D_Uncertainty(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        mid = max(channels // r, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels * 2, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, mean, var=None):
        mean_c = self.pool(mean)
        uncertainty = (
            torch.zeros_like(mean_c)
            if var is None
            else torch.log(var.mean(dim=2, keepdim=True) + 1e-6)
        )
        return self.fc(torch.cat([mean_c, uncertainty], dim=1))


class IAFF_Residual_Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(ch)
        self.att = MS_CAM_1D_Uncertainty(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        a = self.att(x + y, None)
        return self.act(a * y + (1 - a) * x)


class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_ch, out_ch, blocks):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[IAFF_Residual_Block(out_ch) for _ in range(blocks)]
        )

    def forward(self, x):
        return self.blocks(self.head(x))


def sigmoid_probit(mu, var):
    return torch.sigmoid(mu / torch.sqrt(1 + math.pi * var / 8))


class EPResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, m, v):
        m1 = self.fc1(m)
        v1 = v @ (self.fc1.weight ** 2).t() + 1e-6
        p1 = sigmoid_probit(m1, v1)
        v2 = (p1 * (1 - p1)) @ (self.fc2.weight ** 2).t() + 1e-6
        return m + p1, v + v2


class EPResSNN(nn.Module):
    def __init__(self, d_in, d_h, n_cls, blocks):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.blocks = nn.ModuleList([EPResBlock(d_h) for _ in range(blocks)])
        self.fc = nn.Linear(d_h, n_cls)

    def forward_sequence(self, x, eps_var=1e-4):
        batch, steps, dim = x.shape
        m = self.proj(x.reshape(-1, dim))
        v = torch.ones_like(m) * eps_var
        m = sigmoid_probit(m, v)
        v = m * (1 - m) + eps_var
        for block in self.blocks:
            m, v = block(m, v)
        return self.fc(m).reshape(batch, steps, -1).sum(dim=1)


class EPSNNQAC(nn.Module):
    def __init__(self, channels, num_classes, config):
        super().__init__()
        self.feat = ConvFeatureExtractor(
            channels, config.conv_features, config.num_res_blocks
        )
        self.ep = EPResSNN(
            config.conv_features, config.ep_hidden, num_classes, config.num_ep_blocks
        )

    def forward(self, x):
        features = self.feat(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.ep.forward_sequence(features)
