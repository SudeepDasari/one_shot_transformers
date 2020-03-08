import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicEmbeddingModel(nn.Module):
    def __init__(self, k=3, batch_norm=True):
        super(BasicEmbeddingModel, self).__init__()
        norm_factory = lambda f: nn.BatchNorm2d(f) if batch_norm else lambda x: x

        self._conv_1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self._conv_1_n = norm_factory(64)

        self._conv_2_1 = nn.Conv2d(64, 128, k, padding=1)
        self._conv_2_1_n = norm_factory(128)
        self._conv_2_2 = nn.Conv2d(128, 128, k, padding=1)
        self._conv_2_2_n = norm_factory(128)
        self._pool_2 = nn.MaxPool2d(2)

        self._conv_3_1 = nn.Conv2d(128, 256, k, padding=1)
        self._conv_3_1_n = norm_factory(256)
        self._conv_3_2 = nn.Conv2d(256, 256, k, padding=1)
        self._conv_3_2_n = norm_factory(256)
        self._pool_3 = nn.MaxPool2d(2)

        self._conv_4_1 = nn.Conv2d(256, 512, k, padding=1)
        self._conv_4_1_n = norm_factory(512)
        self._conv_4_2 = nn.Conv2d(512, 512, k, padding=1)
        self._conv_4_2_n = norm_factory(512)
        self._pool_4 = nn.MaxPool2d(2)

        self._conv_5_1 = nn.Conv3d(512, 512, k, padding=(2, 1, 1))
        self._conv_5_1_n = nn.BatchNorm3d(512) if batch_norm else lambda x: x
        self._conv_5_2 = nn.Conv3d(512, 512, k, padding=(2, 1, 1))
        self._conv_5_2_n = nn.BatchNorm3d(512) if batch_norm else lambda x: x
        assert k % 2 == 1, "context must be odd"
        self._k = k 

        self._fc6_1 = nn.Linear(512, 512)
        self._fc6_1_n = nn.BatchNorm1d(512) if batch_norm else lambda x: x
        self._fc6_2 = nn.Linear(512, 512)
        self._fc6_2_n = nn.BatchNorm1d(512) if batch_norm else lambda x: x

        self._embed = nn.Linear(512, 128)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape((B * T, C, H, W))
        
        x = F.relu(self._conv_1_n(self._conv_1(x)))

        x = F.relu(self._conv_2_1_n(self._conv_2_1(x)))
        x = F.relu(self._conv_2_2_n(self._conv_2_2(x)))
        x = self._pool_2(x)

        x = F.relu(self._conv_3_1_n(self._conv_3_1(x)))
        x = F.relu(self._conv_3_2_n(self._conv_3_2(x)))
        x = self._pool_3(x)

        x = F.relu(self._conv_4_1_n(self._conv_4_1(x)))
        x = F.relu(self._conv_4_2_n(self._conv_4_2(x)))
        x = self._pool_4(x)

        x = torch.transpose(x.reshape((B, T, 512, x.shape[-2], x.shape[-1])), 1, 2)
        x = F.relu(self._conv_5_1_n(self._conv_5_1(x)))[:,:,:-2,:,:]
        x = F.relu(self._conv_5_2_n(self._conv_5_2(x)))[:,:,:-2,:,:]
        x = [x[:, :, max(0, t - self._k // 2): min(T, t + self._k // 2 + 1),:,:] for t in range(T)]
        x = [torch.max(torch.max(torch.max(t, 2)[0], 2)[0], 2)[0][:, None] for t in x]
        x = torch.cat(x, 1)
        
        x = x.reshape((B * T, 512))
        x = F.relu(self._fc6_1_n(self._fc6_1(x)))
        x = F.relu(self._fc6_2_n(self._fc6_2(x)))

        embed = self._embed(x)
        embed = embed.reshape((B, T, 128))

        return embed


class ResNetFeats(nn.Module):
    def __init__(self, depth=False, out_dim=64):
        super(ResNetFeats, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self._features = nn.Sequential(*list(resnet18.children())[:-1])

        in_dim = 512
        self._depth = depth
        if depth:
            self._conv_1 = nn.Conv2d(1, 32, 7, stride=2, padding=3)
            self._norm_1 = nn.BatchNorm2d(32)
            self._pool_1 = nn.MaxPool2d(4)

            self._conv_2_1 = nn.Conv2d(32, 64, 3, padding=1)
            self._norm_2_1 = nn.BatchNorm2d(64)
            self._conv_2_2 = nn.Conv2d(64, 64, 3, padding=1)
            self._norm_2_2 = nn.BatchNorm2d(64)
            self._pool_2 = nn.MaxPool2d(4)

            self._conv_3_1 = nn.Conv2d(64, 128, 3, padding=1)
            self._norm_3_1 = nn.BatchNorm2d(128)
            self._conv_3_2 = nn.Conv2d(128, 128, 3, padding=1)
            self._norm_3_2 = nn.BatchNorm2d(128)
            self._pool_3 = nn.AdaptiveAvgPool2d(1)
            in_dim += 128
        self._out = nn.Linear(in_dim, out_dim)
        self._out_norm = nn.BatchNorm1d(out_dim)
        self._out_dim = out_dim
    
    def forward(self, x, depth=None):
        reshaped = False
        if len(x.shape) == 5:
            reshaped = True
            B, T, C, H, W = x.shape
            x = x.reshape((B * T, C, H, W))

            if self._depth:
                depth = depth.reshape((B * T, 1, H, W))

        out = self._features(x)[:,:,0,0]
        if self._depth:
            depth = self._pool_1(F.relu(self._norm_1(self._conv_1(depth))))
            depth = F.relu(self._norm_2_1(self._conv_2_1(depth)))
            depth = self._pool_2(self._norm_2_2(self._conv_2_2(depth)))
            depth = F.relu(self._norm_3_1(self._conv_3_1(depth)))
            depth = self._pool_3(self._norm_3_2(self._conv_3_2(depth)))
            depth = depth[:,:,0,0]
            out = torch.cat((out, depth), -1)

        out = F.relu(self._out_norm(self._out(out)))
        if reshaped:
            out = out.reshape((B, T, self._out_dim))
        return out
