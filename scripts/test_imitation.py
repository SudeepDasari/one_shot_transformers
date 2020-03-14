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
from train_imitation import ImitationModule
import pickle as pkl


def rollout_bc(policy, env_type, device, height=224, widht=224, horizon=20, depth=False):
    np.random.seed()
    env = get_env(env_type)(has_renderer=False, reward_shaping=False, use_camera_obs=True, camera_depth=depth, camera_height=height, camera_width=width)
    obs = env.reset()
    
    traj = Trajectory()
    traj.append(obs)
    past_obs = []
    for _ in tqdm(range(env.horizon)):
        past_obs.append([np.transpose(resize(obs['image'], (width, height), True), (2, 0, 1))[None]])
        if depth:
            past_obs[-1].append(np.transpose(resize(obs['depth'][:,:,None], (width, height), False), (2, 0, 1))[None])
        if len(past_obs) > horizon:
            past_obs = past_obs[1:]

        policy_in = [torch.from_numpy(np.concatenate([p[0] for p in past_obs], 0)[None]).to(device)]
        if depth:
            policy_in.append(torch.from_numpy(np.concatenate([p[1] for p in past_obs], 0)[None]).to(device))
        with torch.no_grad():
            action = policy(policy_in, n_samples=5)[0]

        for _ in range(8):
            p = -0.4 * (obs['joint_pos'] - action[:7])
            a = np.concatenate((p, [action[-1]]))
            obs, reward, done, info = env.step(a)
            traj.append(obs, reward, done, info, a)
        if reward or done:
            break
    return traj


def _to_uint(torch_img, new_size=96):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))

    torch_img = np.transpose(torch_img, (1, 2, 0))
    torch_img = torch_img * std + mean
    return resize(torch_img * 255, (new_size, new_size), normalize=False).astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('config', type=str)
    parser.add_argument('--env', type=str, default='SawyerPickPlaceCan')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--save_dir', default='.', type=str)
    args = parser.parse_args()
    config = parse_basic_config(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_path, map_location=device)
    model = model.eval()
    model.to(device)
    policy = MixtureDensitySampler(model)
    
    height = config['dataset'].get('height', 224)
    width = config['dataset'].get('height', 224)
    T = config['dataset'].get('T_pair', 20)
    depth = config['embedding'].get('depth', False)
    for i in range(args.N):
        traj = rollout_bc(policy, args.env, device, height, width, T, depth)
        pkl.dump(traj, open('{}/out{}.pkl'.format(args.save_dir, i), 'wb'))
        out = imageio.get_writer('{}/out{}.gif'.format(args.save_dir, i), fps=30)
        for t in traj:
            out.append_data(t['obs']['image'])
        out.close()
