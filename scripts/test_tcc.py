import torch
import datetime
import argparse
from hem.robosuite.controllers.expert_pick_place import get_expert_trajectory
from hem.models import get_model
from hem.util import parse_basic_config, clean_dict
from hem.datasets.util import resize
import torch.nn as nn
import random
import numpy as np
import cv2


def _extract_images(traj, im_dims=(224, 224), frames=15):
    clip = lambda x : int(max(0, min(x, len(traj) - 1)))
    assert frames >= 2
    
    n_per = len(traj) / frames
    chosen = [resize(traj[0]['obs']['image'], im_dims, True)]
    for t in range(1, frames - 1):
        n = random.randint(clip(t * n_per), clip((t + 1) * n_per - 1))
        chosen.append(resize(traj[n]['obs']['image'], im_dims, True))
    chosen.append(resize(traj[len(traj) - 1]['obs']['image'], im_dims, True))
    chosen = np.concatenate([c[None] for c in chosen])
    return np.transpose(chosen, (0, 3, 1, 2)).astype(np.float32)


def _to_uint(torch_img, new_size=96):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))

    torch_img = np.transpose(torch_img, (1, 2, 0))
    torch_img = torch_img * std + mean
    return resize(torch_img * 255, (new_size, new_size), normalize=False).astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_weights', type=str)
    parser.add_argument('env1', type=str)
    parser.add_argument('env2', type=str)
    parser.add_argument('--N_compars', type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model_weights, map_location=device)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = model.eval()
    model = model.to(device)

    env1_traj = _extract_images(get_expert_trajectory(args.env1))[None]
    env2_trajs = [_extract_images(get_expert_trajectory(args.env2))[None] for _ in range(args.N_compars)]
    
    images = np.concatenate([env1_traj] + env2_trajs, 0)
    with torch.no_grad():
        embeds = model(torch.from_numpy(images).to(device)).cpu().numpy()

    env1_vid, env2_vids = images[0], images[1:]
    env1_embed = embeds[0]
    env2_embeds = embeds[1:]

    all_images = []
    for t in range(env1_embed.shape[0]):
        deltas = np.sum(np.square(env1_embed[t][None,None] - env2_embeds), -1)
        order = deltas.reshape(-1).argsort()

        img_t = _to_uint(env1_vid[t])
        for o in order[:10]:
            traj, traj_t = int(o // env1_embed.shape[0]), o % env1_embed.shape[0]
            img_t = np.concatenate((img_t, _to_uint(env2_vids[traj, traj_t])), 0)
        all_images.append(img_t)
        if t != env1_embed.shape[0] - 1:
            all_images.append(np.zeros((img_t.shape[0], 3, 3), dtype=np.uint8))
    cv2.imwrite('compars.jpg', np.concatenate(all_images, 1)[:,:,::-1])
