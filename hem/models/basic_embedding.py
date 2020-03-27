import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicEmbeddingModel(nn.Module):
    def __init__(self, k=3, batch_norm=True):
        super(BasicEmbeddingModel, self).__init__()
        norm_factory = lambda f: nn.InstanceNorm2d(f, affine=True) if batch_norm else lambda x: x

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
        self._conv_5_1_n = nn.InstanceNorm3d(512, affine=True) if batch_norm else lambda x: x
        self._conv_5_2 = nn.Conv3d(512, 512, k, padding=(2, 1, 1))
        self._conv_5_2_n = nn.InstanceNorm3d(512, affine=True) if batch_norm else lambda x: x
        assert k % 2 == 1, "context must be odd"
        self._k = k 

        self._fc6_1 = nn.Linear(512, 512)
        self._fc6_2 = nn.Linear(512, 512)
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
        x = F.relu(self._fc6_1(x))
        x = F.relu(self._fc6_2(x))

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


class CoordConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding=1, stride=1):
        super().__init__()
        self._conv = nn.Conv2d(in_dim + 2, out_dim, kernel_size, stride, padding)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h_pad = torch.linspace(-1, 1, H).reshape((1, 1, H, 1)).repeat((B, 1, 1, W))
        w_pad = torch.linspace(-1, 1, W).reshape((1, 1, 1, W)).repeat((B, 1, H, 1))
        x = torch.cat((x, h_pad.to(x.device), w_pad.to(x.device)), 1)
        return self._conv(x)


class VGGFeats(nn.Module):
    def __init__(self, out_dim=64, depth=False):
        super().__init__()
        vgg_feats = models.vgg16(pretrained=True).features
        vgg_feats = list(vgg_feats.children())[:4]
        
        c0_in, self._has_depth = 64, False
        if depth:
            self._has_depth = True
            c0_in += 1
        cc0 = CoordConv(c0_in, 64, 3, stride=2)
        n0 = nn.InstanceNorm2d(64, affine=True)
        a0 = nn.ReLU(inplace=True)

        cc1 = CoordConv(64, 128, 3, stride=2)
        n1 = nn.InstanceNorm2d(128, affine=True)
        a1 = nn.ReLU(inplace=True)
        p1 = nn.MaxPool2d(2)

        cc2_1 = CoordConv(128, 256, 3)
        n2_1 = nn.InstanceNorm2d(256, affine=True)
        a2_1 = nn.ReLU(inplace=True)
        cc2_2 = CoordConv(256, 256, 3)
        n2_2 = nn.InstanceNorm2d(256, affine=True)
        a2_2 = nn.ReLU(inplace=True)
        p2 = nn.AdaptiveAvgPool2d(1)

        self._vgg = nn.Sequential(*vgg_feats)
        self._v1 = nn.Sequential(*[cc0, n0, a0])
        self._v2 = nn.Sequential(*[cc1, n1, a1, p1, cc2_1, n2_1, a2_1, cc2_2, n2_2, a2_2, p2])

        self._out_dim = out_dim
        linear = nn.Linear(256, out_dim)
        linear_ac = nn.ReLU(inplace=True)
        self._linear = nn.Sequential(linear, linear_ac)

    def forward(self, x, depth=None):
        if len(x.shape) == 5:
            has_time = True
            B, T, C, H, W = x.shape
            x = x.reshape((B * T, C, H, W))
            if self._has_depth:
                depth = depth.reshape((B * T, 1, H, W))
        else:
            has_time = False
            B, C, H, W = x.shape
        
        if self._has_depth:
            visual_feats = self._vgg(x)
            visual_feats = torch.cat((visual_feats, depth), 1)
            visual_feats = self._v2(self._v1(visual_feats))
        else:
            visual_feats = self._v2(self._v1(self._vgg(x)))
        visual_feats = visual_feats.reshape((visual_feats.shape[0], 256))
        visual_feats = self._linear(visual_feats)

        if has_time:
            return visual_feats.reshape((B, T, self._out_dim))
        return visual_feats
