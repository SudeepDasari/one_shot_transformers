import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


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


class AttentionGoalState(nn.Module):
    def __init__(self, in_dim=2048, out_dim=32, max_len=500, n_heads=8, ntrans=2, dropout=0.1, T=10):
        super().__init__()
        assert in_dim % n_heads == 0, "n_heads must evenly divide input dimension!"
        self._pe = PositionalEncoding(in_dim, max_len=max_len, dropout=dropout)
        self._trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(in_dim, n_heads, dropout=dropout), ntrans)
        self._goal_conv = nn.Conv1d(in_dim, out_dim, T)
        self._mh_attn = nn.MultiheadAttention(in_dim, n_heads)
        self._dpt1, self._dpt2, self._dpt3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self._n1, self._n2 = nn.LayerNorm(in_dim), nn.LayerNorm(out_dim)
        self._ac = nn.ReLU(inplace=True)
        self._l1, self._out = nn.Linear(in_dim, in_dim), nn.Linear(in_dim, out_dim)
        self._out_dim = out_dim

    def forward(self, context, frame):
        """
        context should be [B, T, in_dim]
        frame should be [B, in_dim]
        """
        ctx_embeds = self._pe(context.transpose(0, 1))
        ctx_embeds = self._trans(ctx_embeds)
        goal_embed = torch.mean(self._goal_conv(ctx_embeds.permute((1, 2, 0))), 2)
        
        state_goal_attn = self._mh_attn(frame.transpose(0, 1), ctx_embeds, ctx_embeds)[0].transpose(0, 1)
        state_goal_attn = self._n1(self._dpt1(state_goal_attn) + frame)
        state_goal_attn = self._dpt2(self._ac(self._l1(state_goal_attn))) + frame
        state_goal_attn = self._n2(self._out(state_goal_attn))
        return goal_embed, state_goal_attn
