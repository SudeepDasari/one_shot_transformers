import torch
import torch.nn as nn
import torch.nn.functional as F
from hem.models import get_model
from hem.models.traj_embed import _NonLocalLayer
from hem.models.basic_embedding import _BottleneckConv
from torch.distributions import MultivariateNormal



class CondVAE(nn.Module):
    def __init__(self, inflate, latent_dim, n_non_loc=2, nloc_in=1024, drop_dim=3, dropout=0.1, temp=None):
        super().__init__()
        self._l_dim = latent_dim
        self._embed = get_model('resnet')(output_raw=True, drop_dim=drop_dim)
        self._non_locs = nn.Sequential(*[_NonLocalLayer(nloc_in, nloc_in, dropout=dropout, temperature=temp) for _ in range(n_non_loc)])
        self._temp_pool = nn.Sequential(nn.Conv3d(nloc_in, latent_dim * 2, 3, stride=2), nn.BatchNorm3d(latent_dim*2), nn.ReLU(inplace=True))
        self._spatial_pool = nn.AdaptiveAvgPool2d(latent_dim * 2)
        self._enc_mean, self._enc_ln_var = nn.Linear(latent_dim * 2, latent_dim), nn.Linear(latent_dim * 2, latent_dim)

        self._concat_conv = nn.Sequential(nn.Conv2d(1024 + latent_dim, 1024 + latent_dim, 5, padding=2), 
                                nn.BatchNorm2d(1024 + latent_dim), nn.ReLU(inplace=True))
        self._inflate_layers = []
        last = 1024 + latent_dim
        for d in inflate:
            c_up = nn.ConvTranspose2d(last, d, 2, stride=2)
            c_up_ac = nn.ReLU(inplace=True)
            bottle = _BottleneckConv(d, d, 256)
            self._inflate_layers.extend([c_up, c_up_ac, bottle])
            last = d
        self._inflate_layers = nn.Sequential(*self._inflate_layers)
        self._final = nn.Conv2d(d, 3, 3, padding=1)

    def forward(self, context):
        B = context.shape[0]
        prior = MultivariateNormal(torch.zeros((B, self._l_dim)).to(context.device), torch.diag_embed(torch.ones((B, self._l_dim)).to(context.device)))
        enc_embed = self._embed(context)
        dec_embed = enc_embed[:,0]

        enc_attn = self._non_locs(enc_embed.transpose(1, 2))
        latent_embed = self._spatial_pool(self._temp_pool(enc_attn)[:,:,0])[:,:,0,0]
        mean, var = self._enc_mean(latent_embed), torch.diag_embed(torch.exp(self._enc_ln_var(latent_embed)))
        posterior = MultivariateNormal(mean, var)

        latent_samp = posterior.rsample().unsqueeze(-1).unsqueeze(-1)
        dec_in = torch.cat((dec_embed, latent_samp.repeat((1, 1, dec_embed.shape[2], dec_embed.shape[3]))), 1)
        dec_out = self._final(self._inflate_layers(self._concat_conv(dec_in)))

        kl = torch.distributions.kl_divergence(posterior, prior)
        return dec_out, kl
