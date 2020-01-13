import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicEmbeddingModel(nn.Module):
    def __init__(self, fc1=128, out=32):
        super(BasicEmbeddingModel, self).__init__()
        self._pre_trained = models.resnet18(pretrained=True)
        self._pre_trained = nn.Sequential(*list(self._pre_trained.children())[:-1])

        self._fc1 = nn.Linear(512, fc1)
        self._fc2 = nn.Linear(fc1, out)
    
    def forward(self, x):
        B, T, C, H, W = x.shape

        pt = self._pre_trained(x.reshape((B * T, C, H, W)))
        fc1 = F.relu(self._fc1(torch.squeeze(pt)))
        fc2 = F.relu(self._fc2(fc1))
        return fc2.reshape((B, T, -1))
    