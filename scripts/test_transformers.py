from hem.robosuite.controllers import expert_pick_place
from hem.robosuite import get_env
from hem.robosuite.controllers.expert_pick_place import get_expert_trajectory
from hem.datasets.util import STD, MEAN, select_random_frames, resize, crop
import random
import copy
import os
from hem.util import parse_basic_config
import torch
from hem.datasets import Trajectory
import numpy as np
import pickle as pkl
import imageio
import functools
from hem.models.inverse_module import InverseImitation
from torch.multiprocessing import Pool, set_start_method
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import cv2
import random
set_start_method('forkserver', force=True)


def build_formatter(config):
    def resize_crop(img):
        crop_params = config['dataset'].get('crop', (0,0,0,0))
        height, width = config['dataset'].get('height', 224), config['dataset'].get('width', 224)
        normalize = config['dataset'].get('normalize', True)
        return resize(crop(img, crop_params), (width, height), normalize).transpose((2, 0, 1)).astype(np.float32)
    return resize_crop


def build_env_context(T_context=3, ctr=0):
    task = ctr % 16
    create_seed = random.Random(None)
    create_seed = create_seed.getrandbits(32)
    teacher_expert_rollout = get_expert_trajectory('SawyerPickPlaceDistractor', task=task, seed=create_seed)
    agent_env = get_expert_trajectory('PandaPickPlaceDistractor', task=task, ret_env=True, seed=create_seed)

    context = select_random_frames(teacher_expert_rollout, T_context, True)
    return agent_env, context


def rollout_imitation(model, config, ctr, max_T=60, parallel = False):
    img_formatter = build_formatter(config)
    env, context_frames = build_env_context(config['dataset'].get('T_context', 15), ctr)

    done, states, images = False, [], []
    context = torch.from_numpy(np.concatenate([img_formatter(i)[None] for i in context_frames], 0))[None].cuda()

    np.random.seed(None); env.reset()
    obs = env.reset()
    n_steps = 0
    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False, 'picked': False}
    obj_delta_key = [k for k in obs.keys() if '_to_eef_pos' in k][0]
    obj_key = [k for k in obs.keys() if '0_pos' in k][0]
    start_z = obs[obj_key][2]
    while not done:
        tasks['reached'] =  tasks['reached'] or np.linalg.norm(obs[obj_delta_key][:2]) < 0.03
        tasks['picked'] = tasks['picked'] or (tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate((obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        images.append(img_formatter(obs['image'])[None])

        s_t, i_t = [torch.from_numpy(np.concatenate(arr, 0).astype(np.float32))[None].cuda() for arr in (states, images)]
        with torch.no_grad():
            out = model(s_t, i_t, context)
        
        action = out['bc_distrib'].sample()[0, -1].cpu().numpy()
        action[3:7] = [0.296875, 0.703125, 0.703125, 0.0]
        action[-1] = 1 if action[-1] > 0 and n_steps < max_T - 1 else -1 
        obs, reward, env_done, info = env.step(action)
        
        traj.append(obs, reward, done, info, action)
        tasks['success'] = reward or tasks['success']
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    
    return traj, tasks, context_frames


def _proc(n_workers, model, config, n):
    rollout, task_success_flags, _ = rollout_imitation(model, config, n, parallel = n_workers > 1)
    pkl.dump(rollout, open('./results/traj{}.pkl'.format(n), 'wb'))
    json.dump({k:bool(v) for k, v in task_success_flags.items()}, open('./results/traj{}.json'.format(n), 'w'))
    return task_success_flags


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--config', default='')
    parser.add_argument('--N', default=20, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    config_path = os.path.expanduser(args.config) if args.config else os.path.join(os.path.dirname(model_path), 'config.yaml')
    config = parse_basic_config(config_path, resolve_env=False)

    os.makedirs('./results') if not os.path.exists('./results') else None
    model = InverseImitation(**config['policy'])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')).state_dict())
    model = model.eval().cuda()
    n_success = 0

    f = functools.partial(_proc, args.num_workers, model, config)
    if args.num_workers <= 1:
        task_success_flags = [f(n) for n in range(args.N)]
    else:
        with Pool(args.num_workers) as p:
            task_success_flags = p.map(f, range(args.N))
    
    for k in ['reached', 'picked', 'success']:
        n_success = sum([t[k] for t in task_success_flags])
        print('task {}, rate {}'.format(k, n_success / float(args.N)))
