from hem.models import get_model, Trainer
from hem.models.mdn_loss import MixtureDensityTop, GMMDistribution
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
from hem.models.traj_embed import _NonLocalLayer
from hem.models.contrastive_module import SplitContrastive


class ConditionedPolicy(nn.Module):
    def __init__(self, T_context, action_lstm, context_dim=32, adim=8, sdim=8, n_mixtures=3, agent_latent=8, scene_latent=32, visual_restore=None):
        super().__init__()
        if visual_restore is not None:
            self._embed = torch.load(visual_restore, map_location=torch.device('cpu'))
        else:
            self._embed = SplitContrastive(agent_latent, scene_latent)
        
        vis_dim, hidden_dim = scene_latent + agent_latent, scene_latent + agent_latent + sdim + context_dim
        self._context_proc = nn.Sequential(nn.Linear(T_context * vis_dim, T_context * vis_dim), nn.ReLU(inplace=True), nn.Linear(T_context * vis_dim, context_dim))
        self._in_proc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, action_lstm['in_dim']))
        self._action_lstm = nn.LSTM(action_lstm['in_dim'], action_lstm['out_dim'], action_lstm.get('n_layers', 1))
        self._mdn = MixtureDensityTop(action_lstm['out_dim'], adim, n_mixtures)
    
    def forward(self, states, context):
        context_embed = self.embed_image(context)
        img_embed = self.embed_image(states['images'])

        context_embed = self._context_proc(context_embed.view((context_embed.shape[0], -1))).unsqueeze(1)
        lstm_in = self._in_proc(torch.cat((context_embed.repeat((1, img_embed.shape[1], 1)), img_embed, states['states']), 2))
        lstm_out, _ = self._action_lstm(lstm_in, None)
        mu, sigma_inv, alpha = self._mdn(lstm_out)
        return GMMDistribution(mu, sigma_inv, alpha)
    
    def embed_image(self, image):
        img = image.reshape((image.shape[0] * image.shape[1], image.shape[2], image.shape[3], image.shape[4]))
        embeds = self._embed(img)
        arm_latent = embeds['arm_feat'].view((image.shape[0], image.shape[1], -1))
        scene_latent = embeds['scene_feat'].view((image.shape[0], image.shape[1], -1))
        return torch.cat((arm_latent, scene_latent), -1)


class GoalStateRegression(ConditionedPolicy):
    def __init__(self, T_context, out_dim, agent_latent=8, scene_latent=32, visual_restore=None, constant_embed=False):
        nn.Module.__init__(self)
        if visual_restore is not None:
            self._embed = torch.load(visual_restore, map_location=torch.device('cpu'))
        else:
            assert not constant_embed
            self._embed = SplitContrastive(agent_latent, scene_latent)
        if constant_embed:
            for p in self._embed.parameters():
                p.requires_grad=False

        hidden_dim = (T_context + 1) * (agent_latent + scene_latent)
        self._top = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))

    def forward(self, context, image):
        embeds = self.embed_image(torch.cat((context, image[:,None]), 1))
        embeds = embeds.view((context.shape[0], -1))
        return self._top(embeds)


class _Prior(nn.Module):
    def __init__(self, latent_dim, state_dim, context_dim, T):
        super().__init__()
        self._T = T
        self._context_proc = nn.Sequential(nn.Linear(T * context_dim, T * context_dim), nn.BatchNorm1d(context_dim * T), nn.ReLU(inplace=True), 
                                            nn.Linear(T * context_dim, T * context_dim), nn.BatchNorm1d(context_dim * T), nn.ReLU(inplace=True), 
                                            nn.Linear(T * context_dim, context_dim))
        self._final_proc = nn.Sequential(nn.Linear(context_dim + state_dim, context_dim + state_dim), nn.BatchNorm1d(context_dim + state_dim), nn.ReLU(inplace=True),
                                            nn.Linear(context_dim + state_dim, latent_dim * 2))
        self._l_dim = latent_dim

    def forward(self, s_0, context):
        assert context.shape[1] == self._T, "context times don't match!"
        context_embed = self._context_proc(context.view((context.shape[0], -1)))
        mean_ln_var = self._final_proc(torch.cat((context_embed, s_0), 1))
        mean, ln_var = mean_ln_var[:,:self._l_dim], mean_ln_var[:,self._l_dim:]
        covar = torch.diag_embed(torch.exp(ln_var))
        return MultivariateNormal(mean, covar)


class _Posterior(nn.Module):
    def __init__(self, latent_dim, in_dim, n_layers=1, rnn_dim=128, bidirectional=False):
        super().__init__()
        self._rnn = nn.LSTM(in_dim, rnn_dim, n_layers, bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self._out = nn.Linear(mult * 2 * rnn_dim, latent_dim * 2)
        self._l_dim = latent_dim

    def forward(self, states, actions, lens=None):
        sa = torch.cat((states, actions), 2).transpose(0, 1)
        sa = torch.nn.utils.rnn.pack_padded_sequence(sa, lens, enforce_sorted=False) if lens is not None else sa
        self._rnn.flatten_parameters()
        rnn_out, _ = self._rnn(sa)
        if lens is not None:
            rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out)
            front, back = rnn_out[0], rnn_out[lens-1, range(lens.shape[0])]
        else:
            front, back = rnn_out[0], rnn_out[1]
        mean_ln_var = self._out(torch.cat((front, back), 1))
        mean, ln_var = mean_ln_var[:,:self._l_dim], mean_ln_var[:,self._l_dim:]
        covar = torch.diag_embed(torch.exp(ln_var))
        return MultivariateNormal(mean, covar)


class LatentImitation(nn.Module):
    def __init__(self, config):
        super().__init__()
        # initialize visual embeddings
        embed = get_model(config['image_embedding'].pop('type'))
        self._embed = embed(**config['image_embedding'])

        self._concat_state = config.get('concat_state', True)
        latent_dim = config['latent_dim']
        self._prior = _Prior(latent_dim=latent_dim, **config['prior'])
        self._posterior = _Posterior(latent_dim=latent_dim, **config['posterior'])

        # action processing
        self._action_lstm = nn.LSTM(config['action_lstm']['in_dim'], config['action_lstm']['out_dim'], config['action_lstm'].get('n_layers', 1))
        self._mdn = MixtureDensityTop(config['action_lstm']['out_dim'], config['adim'], config['n_mixtures'])
    
    def forward(self, states, images, context, actions=None, ret_dist=True):
        img_embed = self._embed(images)
        context_embed = self._embed(context)
        states = torch.cat((img_embed, states), 2) if self._concat_state else img_embed

        prior = self._prior(states[:,0], context_embed)
        goal_latent = prior.rsample()
        posterior = self._posterior(states, actions) if actions is not None else prior

        if self.training:
            assert actions is not None
            sa_latent = posterior.rsample()
        else:
            sa_latent = prior.rsample()
        
        lstm_in = torch.cat((sa_latent, goal_latent), 1)[None].repeat((states.shape[1], 1, 1))
        lstm_in = torch.cat((lstm_in, states.transpose(0, 1)), 2)
        self._action_lstm.flatten_parameters()
        pred_embeds, _ = self._action_lstm(lstm_in)
        mu, sigma_inv, alpha = self._mdn(pred_embeds.transpose(0, 1))
        if ret_dist:
            return GMMDistribution(mu, sigma_inv, alpha), (posterior, prior)
        return (mu, sigma_inv, alpha), torch.distributions.kl.kl_divergence(posterior, prior)


class _NormalPrior(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self._l_dim = latent_dim
    
    def forward(self, states, context):
        mean = torch.zeros((states.shape[0], self._l_dim)).to(states.device)
        covar = torch.diag_embed(torch.ones((states.shape[0], self._l_dim))).to(states.device)
        return MultivariateNormal(mean, covar), None


class _SimplePrior(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self._prior_nn = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU(inplace=True), nn.Linear(latent_dim, latent_dim), nn.ReLU(inplace=True))
        self._ln_var = nn.Linear(latent_dim, latent_dim)
        self._mean = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, states, context):
        nn_in = torch.cat((states, context), 1)
        p_latent = self._prior_nn(nn_in)
        mean = self._mean(p_latent)
        covar = torch.diag_embed(torch.exp(self._ln_var(p_latent)))
        return MultivariateNormal(mean, covar), None
 

class _VidPrior(nn.Module):
    def __init__(self, vec_dim, latent_dim, n_nonloc=1, dropout=0.3, T=4, embed_restore='', temp=None, loc_dim=1024, drop_dim=3, bottleneck_dim=14, HW=300, ret_bottleneck=False):
        super().__init__()
        self._embed = get_model('resnet')(output_raw=True, drop_dim=drop_dim)
        if embed_restore:
            embed_restore = torch.load(embed_restore, map_location=torch.device('cpu')).state_dict()
            self._embed.load_state_dict(embed_restore, strict=False)
        self._nonlocs = nn.Sequential(*[_NonLocalLayer(loc_dim, loc_dim, temperature=temp, dropout=dropout) for _ in range(n_nonloc)])
        self._project_channel = nn.Linear(loc_dim, 1)
        self._to_bottlekneck = nn.Sequential(nn.Linear(HW * T, bottleneck_dim), nn.ReLU(inplace=True), nn.Linear(bottleneck_dim, bottleneck_dim))

        self._latent_proc = nn.Sequential(nn.Linear(bottleneck_dim + vec_dim, latent_dim), nn.ReLU(inplace=True))
        self._mean, self._ln_var = nn.Linear(latent_dim, latent_dim), nn.Linear(latent_dim, latent_dim)
        self._ret_bottleneck = ret_bottleneck

    def forward(self, states, context):
        context_embeds = self._embed(context)
        context_embeds = self._nonlocs(context_embeds.transpose(1, 2))
        proj_embeds = self._project_channel(context_embeds.view((context_embeds.shape[0], context_embeds.shape[1], -1)).transpose(1, 2))[:,:,0]
        bottleneck = self._to_bottlekneck(proj_embeds)

        latent = self._latent_proc(torch.cat((bottleneck, states), 1))
        mean, covar = self._mean(latent), torch.diag_embed(torch.exp(self._ln_var(latent)))

        if self._ret_bottleneck:
            return MultivariateNormal(mean, covar), bottleneck
        return MultivariateNormal(mean, covar), None


class LatentStateImitation(nn.Module):
    def __init__(self, adim, sdim, n_layers, latent_dim, lstm_dim=32, post_rnn_dim=64, post_rnn_layers=1, n_mixtures=3, post_bidirect=True, ss_dim=7,ss_ramp=10000, aux_dim=0, prior_dim=0, append_prior=False, aux_on_prior=False, vid_prior=None):
        super().__init__()
        if vid_prior is not None:
            self._prior = _VidPrior(**vid_prior)
        else:
            self._prior = _SimplePrior(prior_dim, latent_dim) if prior_dim else _NormalPrior(latent_dim=latent_dim)
        self._posterior = _Posterior(latent_dim, sdim + adim, n_layers=post_rnn_layers, rnn_dim=post_rnn_dim, bidirectional=post_bidirect)

        # action processing
        self._append_prior = append_prior
        self._aux_on_prior = aux_on_prior
        mult = 2 if append_prior else 1
        self._action_lstm = nn.LSTM(sdim + mult * latent_dim, lstm_dim, n_layers)
        self._mdn = MixtureDensityTop(lstm_dim, adim, n_mixtures) if n_mixtures else None
        if not n_mixtures:
            self._ac_top = nn.Linear(lstm_dim, adim)
        assert ss_ramp > 0, 'must be non neg'
        self._t, self._ramp, self._ss_dim = 0, ss_ramp, ss_dim
        self._aux = nn.Sequential(nn.Linear(mult * latent_dim, latent_dim), nn.ReLU(inplace=True), nn.Linear(latent_dim, aux_dim)) if aux_dim else None

    def forward(self, states, actions=None, lens=None, ret_dist=True, force_ss=False, force_no_ss=False, prev_latent=None, context=None, sample_prior=False):
        assert not force_ss or not force_no_ss, "both settings cannot be true!"
        
        if prev_latent is not None:
            posterior, prior = None, None
            prior_sample = sa_latent =  prev_latent
        else:
            prior, prior_aux = self._prior(states[:,0], context)
            prior_sample = prior.rsample()
            posterior = self._posterior(states, actions, lens=lens) if actions is not None else prior
            sa_latent = prior.rsample() if sample_prior is None else posterior.rsample()
        
        if self._append_prior:
            sa_latent = torch.cat((sa_latent, prior_sample), 1)

        self._action_lstm.flatten_parameters()
        hidden, ss_p = None, self.ss_p
        pred, last_acs = [], None
        for t in range(states.shape[1]):
            if t == 0 or (not self.training and not force_ss) or force_no_ss:
                in_t = torch.cat((states[:,t], sa_latent), 1)
            else:
                select = [np.random.choice(2, p=[1-ss_p, ss_p]) for _ in range(states.shape[0])]
                prefixes = torch.cat((states[:,t,:self._ss_dim][None], last_acs[None,:,:self._ss_dim]), 0)
                prefixes = prefixes[select, range(states.shape[0])]
                in_state = torch.cat((prefixes, states[:,t,self._ss_dim:]), 1)
                in_t = torch.cat((in_state, sa_latent), 1)

            lstm_out, hidden = self._action_lstm(in_t[None], hidden)
            if self._mdn is not None:
                mu_t, sigma_t, alpha_t = self._mdn(lstm_out)
                pred.append((mu_t.transpose(0, 1), sigma_t.transpose(0, 1), alpha_t.transpose(0, 1)))
                if self.training or force_ss:
                    dist_t = GMMDistribution(mu_t[0].detach(), sigma_t[0].detach(), alpha_t[0].detach())
                    last_acs = dist_t.sample()
            else:
                a_out = self._ac_top(lstm_out)
                pred.append(a_out.transpose(0, 1))
                last_acs = a_out.detach()[0]
        
        if prior_aux is not None:
            aux_pred = prior_aux
        else:
            aux_in = sa_latent if not self._aux_on_prior else prior_sample
            aux_pred = self._aux(aux_in) if self._aux is not None else None

        if self._mdn is not None:
            mu, sigma_inv, alpha = [torch.cat([tens[j] for tens in pred], 1) for j in range(3)]
            if ret_dist:
                return GMMDistribution(mu, sigma_inv, alpha), (posterior, prior), sa_latent
            return (mu, sigma_inv, alpha), (torch.distributions.kl.kl_divergence(posterior, prior), aux_pred)

        pred_acs = torch.cat(pred, 1)
        if ret_dist:
            return pred_acs, (posterior, prior), sa_latent
        return pred_acs, (torch.distributions.kl.kl_divergence(posterior, prior), aux_pred)

    def replace_prior(self, other):
        self._prior = other

    @property
    def ss_p(self):
        return min(self._t / float(self._ramp), 1)

    def increment_ss(self):
        self._t += 1
