import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np


class BaseTraj(nn.Module):
    def __init__(self, latent_dim=256, max_len=500, nhead=8, ntrans=3, dropout=0):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        self._resnet_features = nn.Sequential(*list(resnet50.children())[:-1])
        self._pe = PositionalEncoding(2048, max_len=max_len, dropout=dropout)
        self._trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(2048, nhead, dropout=dropout), ntrans)
        self._out = nn.Linear(2048, latent_dim)
        self._latent_dim = latent_dim

    def forward(self, vids):
        B, T, C, H, W = vids.shape
        imgs = vids.reshape((B * T, C, H, W))
        embeds = self._resnet_features(imgs).reshape((B, T, 2048))
        embeds = self._pe(embeds.transpose(0, 1))
        embeds = self._trans(embeds).transpose(0, 1)
        embeds = torch.mean(embeds, 1)
        return self._out(embeds)

    @property
    def dim(self):
        return self._latent_dim


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding from Pytorch Seq2Seq Documentation
    source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class _NonLocalLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feedforward_dim=512, dropout=0):
        super().__init__()
        self._in_dim = in_dim
        self._K = nn.Conv3d(in_dim, feedforward_dim, 1)
        self._V = nn.Conv3d(in_dim, feedforward_dim, 1)
        self._Q = nn.Conv3d(in_dim, feedforward_dim, 1)
        self._a1, self._drp1 = nn.ReLU(inplace=dropout==0), nn.Dropout3d(dropout)
        self._out = nn.Conv3d(feedforward_dim, out_dim, 1)
        self._a2, self._drop2 = nn.ReLU(inplace=dropout==0), nn.Dropout3d(dropout)
        self._norm = nn.BatchNorm3d(out_dim)
    
    def forward(self, inputs):
        K, Q, V = self._K(inputs), self._Q(inputs), self._V(inputs)
        B, C, T, H, W = K.shape
        
        KQ = torch.matmul(K.reshape((B, C, T*H*W)).transpose(1, 2), Q.reshape((B, C, T*H*W)))
        attn = F.softmax(KQ / np.sqrt(self._in_dim), 2)
        V = torch.sum(V.view((B, C, T*H*W, 1)) * attn[:,None], 3).reshape((B, C, T, H, W))
        V = self._drp1(self._a1(V))
        return self._norm(inputs + self._drop2(self._a2(self._out(V))))

class AttentionGoalState(nn.Module):
    def __init__(self, in_dim=1024, out_dim=128, max_pos_len=3000, num_attn=2, dropout=0.0, T=3, ff_dim=512):
        super().__init__()
        self._pe = PositionalEncoding(out_dim, max_len=max_pos_len, dropout=dropout)
        self._nonloc_layers = nn.Sequential(*[_NonLocalLayer(in_dim, in_dim, ff_dim, dropout) for _ in range(num_attn)])
        self._temporal_pool = nn.Sequential(nn.Conv3d(in_dim, out_dim, T), nn.ReLU(inplace=True))
        self._spatial_pool = nn.AdaptiveAvgPool3d(1)
        upsample_goal = [torch.nn.ConvTranspose2d(in_dim, ff_dim, 2, stride=2), nn.ReLU(inplace=True)]
        upsample_goal.append(torch.nn.ConvTranspose2d(ff_dim, out_dim, 2, stride=2))
        self._upsample_goal = nn.Sequential(*upsample_goal)
        self._out_dim = out_dim

    def forward(self, context, frame):
        """
        context should be [B, T, in_dim]
        frame should be [B, in_dim]
        """
        ctx_embeds = self._nonloc_layers(context.transpose(1, 2))
        goal_embed = self._spatial_pool(self._temporal_pool(ctx_embeds))[:,:,0,0,0]
        
        B, T, C, H, W = frame.shape
        fr = frame.view((B * T, C, H, W))
        state_embed = self._upsample_goal(frame.view((B * T, C, H, W))).reshape((B, T, self._out_dim, H*4, W*4))
        state_embed = state_embed.reshape((B * T, self._out_dim, H * W * 16))
        V = self._pe(state_embed.permute(2, 0, 1)).transpose(0, 1)
        
        SG = torch.matmul(state_embed.transpose(1, 2), goal_embed[:,:,None].repeat((T, 1, 1))) / np.sqrt(self._out_dim)
        state_goal_attn = torch.sum(torch.softmax(SG, 1) * V, 1)
        return goal_embed, state_goal_attn.reshape((B, T, self._out_dim))


class GoalState(nn.Module):
    def __init__(self, in_dim=2048, hidden=[256, 128], out_dim=64, T=5):
        super().__init__()
        goal_module = []
        last_in = in_dim
        for  d in hidden + [out_dim]:
            l, a, n = nn.Linear(last_in, d), nn.ReLU(inplace=True), nn.BatchNorm1d(d)
            goal_module.extend([l, a, n])
            last_in = d
        self._goal_module = nn.ModuleList(goal_module)
        self._goal_conv = nn.Conv1d(out_dim, out_dim, T)

        state_module = []
        last_in = in_dim
        for  d in hidden + [out_dim]:
            l, a, n = nn.Linear(last_in, d), nn.ReLU(inplace=True), nn.BatchNorm1d(d)
            state_module.extend([l, a, n])
            last_in = d
        self._state_module = nn.ModuleList(state_module)

    def forward(self, context, frame):
        ctx_embeds = self._apply_module(self._goal_module, context)
        ctx_embeds = F.relu(torch.mean(self._goal_conv(ctx_embeds.transpose(1, 2)), dim=-1))
        return ctx_embeds, self._apply_module(self._state_module, frame)
    
    def _apply_module(self, module, embeds):
        embeds = embeds
        for l in range(int(len(module) / 3)):
            l, a, n = module[3*l:3*l + 3]
            embeds = n(a(l(embeds)).transpose(1, 2)).transpose(1, 2)
        return embeds
