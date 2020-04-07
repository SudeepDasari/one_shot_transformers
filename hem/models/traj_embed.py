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
