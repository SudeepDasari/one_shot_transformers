import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from hem.models.traj_embed import _NonLocalLayer


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
    def __init__(self, out_dim=256, output_raw=False, drop_dim=1):
        super(ResNetFeats, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self._features = nn.Sequential(*list(resnet50.children())[:-drop_dim])
        self._output_raw = output_raw
        if not output_raw:
            self._out = nn.Sequential(nn.Linear(2048, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim))
            self._out_dim = out_dim
    
    def forward(self, x, depth=None):
        reshaped = False
        if len(x.shape) == 5:
            reshaped = True
            B, T, C, H, W = x.shape
            x = x.reshape((B * T, C, H, W))
        out = self._features(x)
        NH, NW = out.shape[-2:]
        
        # reshape to proper dimension
        if reshaped:
            out = out.reshape((B, T, -1, NH, NW))
        if NH * NW == 1:
            if reshaped:
                out = out[:,:,:,0,0]
            else:
                out = out[:,:,0,0]

        if not self._output_raw:
            assert len(out.shape) < 4, "only works on non-spatial input!"
            return self._out(out)
        return out

    @property
    def dim(self):
        return self._out_dim


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

        cc1 = nn.Conv2d(64, 64, 3, 2, 1)
        n1 = nn.InstanceNorm2d(64, affine=True)
        a1 = nn.ReLU(inplace=True)
        p1 = nn.MaxPool2d(2)

        cc2_1 = nn.Conv2d(64, 64, 3, 1, 1)
        n2_1 = nn.InstanceNorm2d(64, affine=True)
        a2_1 = nn.ReLU(inplace=True)
        cc2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        n2_2 = nn.InstanceNorm2d(64, affine=True)
        a2_2 = nn.ReLU(inplace=True)
        p2 = nn.AdaptiveAvgPool2d(1)

        self._vgg = nn.Sequential(*vgg_feats)
        self._v1 = nn.Sequential(*[cc0, n0, a0])
        self._v2 = nn.Sequential(*[cc1, n1, a1, p1, cc2_1, n2_1, a2_1, cc2_2, n2_2, a2_2, p2])

        self._out_dim = out_dim
        linear = nn.Linear(64, out_dim)
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
        visual_feats = visual_feats.reshape((visual_feats.shape[0], 64))
        visual_feats = self._linear(visual_feats)

        if has_time:
            return visual_feats.reshape((B, T, self._out_dim))
        return visual_feats

    @property
    def dim(self):
        return self._out_dim


class _BottleneckConv(nn.Module):
    def __init__(self, in_dim, out_dim, feed_forward):
        super().__init__()
        self._c1 = nn.Conv2d(in_dim, feed_forward, 1, stride=1, padding=0)
        self._a1 = nn.ReLU(inplace=True)
        self._c2 = nn.Conv2d(feed_forward, feed_forward, 3, stride=1, padding=1)
        self._a2 = nn.ReLU(inplace=True)
        self._c3 = nn.Conv2d(feed_forward, out_dim, 1, stride=1, padding=0)
        self._n = nn.InstanceNorm2d(out_dim)
        self._residual = in_dim == out_dim

    def forward(self, x):
        inter = self._c3(self._a2(self._c2(self._a1(self._c1(x)))))
        if self._residual:
            return self._n(x + inter)
        return self._n(x)


class SimpleSpatialSoftmax(nn.Module):
    def __init__(self, added_convs=3, n_out=128):
        super().__init__()
        vgg_feats =list( models.vgg16(pretrained=True).features[:16].children())
        added_feats = [_BottleneckConv(256, 256, 64) for _ in range(added_convs)]
        self._conv_stack = nn.Sequential(*(vgg_feats + added_feats))
        self._out = nn.Sequential(nn.Linear(512, 512), nn.ReLU(512), nn.Linear(512, n_out))

    def forward(self, x):
        reshaped = len(x.shape) == 5
        if reshaped:
            B, T, C, H, W = x.shape
            x = x.view((B * T, C, H, W))
        x = self._conv_stack(x)
        x = F.softmax(x.view((B*T, x.shape[1], -1)), dim=2).view((B*T, x.shape[1], x.shape[2], x.shape[3]))
        h = torch.sum(torch.linspace(-1, 1, x.shape[2]).view((1, 1, -1)).to(x.device) * torch.sum(x, 3), 2)
        w = torch.sum(torch.linspace(-1, 1, x.shape[3]).view((1, 1, -1)).to(x.device) * torch.sum(x, 2), 2)
        x = torch.cat((h, w), 1)
        if reshaped:
            x = x.view((B, T, 512))
        return self._out(x)


class AuxModel(nn.Module):
    def __init__(self, bottleneck_dim, n_nonloc=2, dropout=0.1, T=4, embed_restore='', temp=None, loc_dim=1024, drop_dim=3, HW=300):
        super().__init__()
        self._embed = ResNetFeats(output_raw=True, drop_dim=drop_dim)
        if embed_restore:
            embed_restore = torch.load(embed_restore, map_location=torch.device('cpu')).state_dict()
            self._embed.load_state_dict(embed_restore, strict=False)
        self._nonlocs = nn.Sequential(*[_NonLocalLayer(loc_dim, loc_dim, temperature=temp, dropout=dropout) for _ in range(n_nonloc)])
        self._project_channel = nn.Linear(loc_dim, 1)
        self._to_bottlekneck = nn.Sequential(nn.Linear(HW * T, bottleneck_dim), nn.ReLU(inplace=True), nn.Linear(bottleneck_dim, bottleneck_dim))
        
    def forward(self, context):
        context_embeds = self._embed(context)
        context_embeds = self._nonlocs(context_embeds.transpose(1, 2))
        proj_embeds = self._project_channel(context_embeds.view((context_embeds.shape[0], context_embeds.shape[1], -1)).transpose(1, 2))[:,:,0]
        bottleneck = self._to_bottlekneck(proj_embeds)
        return bottleneck
