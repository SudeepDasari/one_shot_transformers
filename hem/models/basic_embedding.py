import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicEmbeddingModel(nn.Module):
    def __init__(self, k=3):
        super(BasicEmbeddingModel, self).__init__()
        self._conv_1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)

        self._conv_2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self._conv_2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self._pool_2 = nn.MaxPool2d(2)

        self._conv_3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self._conv_3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self._pool_3 = nn.MaxPool2d(2)

        self._conv_4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self._conv_4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self._pool_4 = nn.MaxPool2d(2)

        self._conv_5_1 = nn.Conv3d(512, 512, 3, padding=1)
        self._conv_5_2 = nn.Conv3d(512, 512, 3, padding=1)
        assert k % 2 == 1, "context must be odd"
        self._k = k 

        self._fc6_1 = nn.Linear(512, 512)
        self._fc6_2 = nn.Linear(512, 512)

        self._embed = nn.Linear(512, 128)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape((B * T, C, H, W))
        
        x = F.relu(self._conv_1(x))

        x = F.relu(self._conv_2_1(x))
        x = F.relu(self._conv_2_2(x))
        x = self._pool_2(x)

        x = F.relu(self._conv_3_1(x))
        x = F.relu(self._conv_3_2(x))
        x = self._pool_3(x)

        x = F.relu(self._conv_4_1(x))
        x = F.relu(self._conv_4_2(x))
        x = self._pool_4(x)

        x = torch.transpose(x.reshape((B, T, 512, x.shape[-2], x.shape[-1])), 1, 2)
        x = F.relu(self._conv_5_1(x))
        x = F.relu(self._conv_5_2(x))
        x = [x[:, :, max(0, t - self._k // 2): min(T, t + self._k // 2 + 1),:,:] for t in range(T)]
        x = [torch.max(torch.max(torch.max(t, 2)[0], 2)[0], 2)[0][:, None] for t in x]
        x = torch.cat(x, 1)
        
        x = x.reshape((B * T, 512))
        x = F.relu(self._fc6_1(x))
        x = F.relu(self._fc6_2(x))

        embed = self._embed(x)
        embed = embed.reshape((B, T, 128))

        return embed
