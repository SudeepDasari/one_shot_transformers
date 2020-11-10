import torch
import torch.nn as nn
import torch.nn.functional as F
from hem.models import get_model
from torchvision import models
from hem.models.inverse_module import _DiscreteLogHead
from hem.models.discrete_logistic import DiscreteMixLogistic


class ContextualLSTM(nn.Module):
    def __init__(self, latent_dim, lstm_out=32, n_mix=2, adim=8, const_var=True, n_lstm_layer=1, dropout=0.2, embed_hidden=256, attn_temp=None, use_attn=True):
        super().__init__()
        self._resnet18 = get_model('resnet')(output_raw=True, drop_dim=2, use_resnet18=True)
        self._to_embed = nn.Sequential(nn.Linear(1024, embed_hidden), nn.Dropout(dropout), nn.ReLU(), nn.Linear(embed_hidden, latent_dim))
        self._encoder = nn.LSTM(latent_dim, lstm_out, n_lstm_layer)
        self._decoder = nn.LSTM(latent_dim, lstm_out, n_lstm_layer)
        self._hidden_2_attn = nn.Sequential(nn.Linear(2 * lstm_out, 2 * lstm_out), nn.ReLU(), nn.Linear(2 * lstm_out, latent_dim))
        self._action_dist = _DiscreteLogHead(lstm_out, adim, n_mix, const_var)
        self._temp = attn_temp if attn_temp else 1
        self._use_attn = use_attn

    def forward(self, states, images, context, ret_dist=True):
        context_embed = self._embed_images(context)
        img_embed = self._embed_images(images)
        
        self._encoder.flatten_parameters()
        self._decoder.flatten_parameters()
        _, hidden = self._encoder(context_embed.transpose(0, 1))
        if self._use_attn:
            dec_out = []
            for t in range(img_embed.shape[1]):
                all_prev = torch.cat((context_embed, img_embed[:,:t+1]), dim=1)
                attn_query = self._hidden_2_attn(torch.cat((hidden[0][0], hidden[1][0]), 1))
                attn_weights = torch.matmul(attn_query[:,None], all_prev.transpose(1, 2)) / self._temp
                attn_weights = F.softmax(attn_weights, dim=2)
                
                dec_in = torch.sum(attn_weights.transpose(1, 2) * all_prev, dim=1)
                out, hidden = self._decoder(dec_in[None], hidden)
                dec_out.append(out)
            dec_out = torch.cat(dec_out, 0).transpose(0, 1)
        else:
            dec_out, _ = self._decoder(img_embed.transpose(0, 1), hidden)
            dec_out = dec_out.transpose(0, 1)
        mu, scale, logit = self._action_dist(dec_out)
        out = DiscreteMixLogistic(mu, scale, logit) if ret_dist else (mu, scale, logit)
        return out

    def _embed_images(self, images):
        features = self._resnet18(images)
        features = F.softmax(features.reshape((features.shape[0], features.shape[1], features.shape[2], -1)), dim=3).reshape(features.shape)
        h = torch.sum(torch.linspace(-1, 1, features.shape[3]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 4), 3)
        w = torch.sum(torch.linspace(-1, 1, features.shape[4]).view((1, 1, 1, -1)).to(features.device) * torch.sum(features, 3), 3)
        return F.normalize(self._to_embed(torch.cat((h, w), 2)), dim=2)


class DAMLNetwork(nn.Module):
    def __init__(self, n_final_convs, aux_dim=6, adim=8, n_mix=2, T_context=2, const_var=True, maml_lr=None, first_order=None):
        super().__init__()
        if n_final_convs == 'resnet':
            self._is_resnet = True
            self._visual_net = get_model('resnet')(output_raw=True, drop_dim=2, use_resnet18=True)
            self._resnet_2_embed = nn.Sequential(nn.Linear(1024, 256), nn.Dropout(p=0.2), nn.ReLU(), nn.Linear(256, 256))
            n_final_convs = 128
        else:
            self._is_resnet = False
            vgg_pre = models.vgg16(pretrained=True).features[:5]
            c3, a3 = nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU()
            c4, a4 = nn.Conv2d(64, n_final_convs, 3, stride=2, padding=1), nn.ReLU()
            c5, a5 = nn.Conv2d(n_final_convs, n_final_convs, 3, stride=1, padding=1), nn.ReLU()
            self._visual_net = nn.Sequential(vgg_pre, c3, a3, c4, a4, c5, a5)
        self._aux_pose = nn.Sequential(nn.Linear(n_final_convs * 2, 32), nn.ReLU(), nn.Linear(32, aux_dim))
        self._action_mlp = nn.Sequential(nn.Linear(n_final_convs * 2, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU())
        self._action_dist = _DiscreteLogHead(50, adim, n_mix, const_var)
        dim1, dim2 = T_context * (50 + 2 * n_final_convs), 50 * T_context
        self._learned_loss = nn.Sequential(nn.Linear(dim1, dim2), nn.ReLU(), nn.LayerNorm(dim2), nn.Linear(dim2, dim2), nn.ReLU(), nn.LayerNorm(dim2), nn.Linear(dim2, 1))

    def forward(self, states, images, ret_dist=True, learned_loss=False):
        img_embed = self._embed_images(images)
        action_mlp = self._action_mlp(img_embed)

        out = {}
        if learned_loss:
            learned_in = torch.cat((action_mlp, img_embed), -1)
            out['learned_loss'] = torch.squeeze(self._learned_loss(learned_in.reshape((1, -1))))
        else:
            mu, sigma_inv, alpha = [x[0] for x in self._action_dist(action_mlp[None])]
            out['action_dist'] = DiscreteMixLogistic(mu, sigma_inv, alpha) if ret_dist else (mu, sigma_inv, alpha)
            out['aux'] = self._aux_pose(img_embed[:1])
        return out

    def _embed_images(self, images):
        features = self._visual_net(images)
        features = F.softmax(features.reshape((features.shape[0], features.shape[1], -1)), dim=2).reshape(features.shape)
        h = torch.sum(torch.linspace(-1, 1, features.shape[2]).view((1, 1, -1)).to(features.device) * torch.sum(features, 3), 2)
        w = torch.sum(torch.linspace(-1, 1, features.shape[3]).view((1, 1, -1)).to(features.device) * torch.sum(features, 2), 2)
        spatial_softmax = torch.cat((h, w), 1)
        if self._is_resnet:
            return self._resnet_2_embed(spatial_softmax)
        return spatial_softmax