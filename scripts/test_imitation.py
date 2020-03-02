import torch
import datetime
import argparse
from hem.models import get_model
from hem.models.mdn_loss import MixtureDensityTop, MixtureDensitySampler
from hem.util import parse_basic_config, clean_dict
from hem.datasets.util import resize
import torch.nn as nn
import random
import numpy as np
import cv2
from hem.datasets import Trajectory
from hem.robosuite import get_env
import imageio
from tqdm import tqdm


def rollout_bc(policy, env_type, device, N=10, height=224, widht=224, horizon=20):
    trajs = []
    for _ in range(N):
        np.random.seed()
        env = get_env(env_type)(has_renderer=False, reward_shaping=False, use_camera_obs=True)
        obs = env.reset()
        
        traj = Trajectory()
        traj.append(obs)
        past_obs = []
        for _ in tqdm(range(env.horizon)):
            past_obs.append(np.transpose(resize(obs['image'], (width, height), True), (2, 0, 1))[None])
            if len(past_obs) > horizon:
                past_obs = past_obs[1:]
            
            policy_in = torch.from_numpy(np.concatenate(past_obs, 0)[None]).to(device)
            with torch.no_grad():
                action = policy(policy_in)[0]
            
            obs, reward, done, info = env.step(action)
            traj.append(obs, reward, done, info, action)
            if reward or done:
                break
        trajs.append(traj)
    return trajs


def _to_uint(torch_img, new_size=96):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))

    torch_img = np.transpose(torch_img, (1, 2, 0))
    torch_img = torch_img * std + mean
    return resize(torch_img * 255, (new_size, new_size), normalize=False).astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_config', type=str)
    parser.add_argument('model_weights', type=str)
    parser.add_argument('--env', type=str, default='SawyerPickPlaceCan')
    parser.add_argument('--N', type=int, default=10)
    args = parser.parse_args()
    config = parse_basic_config(args.model_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_class = get_model(config['model'].pop('type'))
    base_model = model_class(**config['model'])
    mdn = MixtureDensityTop(**config['mdn'])
    model = nn.Sequential(base_model, mdn)
    model.load_state_dict(clean_dict(torch.load(args.model_weights, map_location=device)))
    model.to(device)
    model.eval()
    
    height = config['dataset'].get('height', 224)
    width = config['dataset'].get('height', 224)
    T = config['dataset'].get('T_pair', 20)
    rollouts = rollout_bc(MixtureDensitySampler(model), args.env, device, args.N, height, width, T)

    for i, traj in enumerate(rollouts):
        out = imageio.get_writer('out{}.gif'.format(i))
        for t in traj:
            out.append_data(t['obs']['image'])
        out.close()
