import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaseRNN(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=3, batch_first=True, visual_input=False):
        super().__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._n_layers = n_layers
        self._batch_first = batch_first
        self._visual_input = visual_input

        if visual_input:
            resnet18 = models.resnet18(pretrained=True)
            self._features = nn.Sequential(*list(resnet18.children())[:-1])
            assert in_dim == 512, "default visual features have dimension 512!"

        self._gru = nn.GRU(in_dim, out_dim, n_layers, batch_first=batch_first)
    
    def forward(self, x):
        if self._visual_input:
            if self._batch_first:
                B, L, C, H, W = x.shape
            else:
                L, B, C, H, W = x.shape
            x = torch.squeeze(self._features(x.reshape((B * L, C, H, W))))

            if self._batch_first:
                x = x.reshape((B, L, self._in_dim))
            else:
                x = x.reshape((L, B, self._in_dim))
        self._gru.flatten_parameters()
        return self._gru(x)[0]


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, k=15, BTC=True, norm=True):
        super().__init__()
        self._conv = nn.Conv1d(in_dim, out_dim, k, padding=k-1)
        self._k = k
        self._norm = lambda x: x
        if norm:
            self._norm_layer = nn.LayerNorm(out_dim)
            self._norm = lambda x: torch.transpose(self._norm_layer(torch.transpose(x, 1, 2)), 1, 2)
        self._ac = nn.ReLU(inplace=True)
        self._BTC = BTC
    
    def forward(self, x):
        if self._BTC:
            x = torch.transpose(x, 1, 2)
        x = self._conv(x)[:,:,:-(self._k-1)]
        x = self._ac(self._norm(x))
        if self._BTC:
            x = torch.transpose(x, 1, 2)
        return x
