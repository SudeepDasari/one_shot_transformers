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
