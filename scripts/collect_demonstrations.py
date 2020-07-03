from hem.robosuite.controllers.expert_pick_place import get_expert_trajectory
from hem.robosuite.controllers.random_reach import get_random_trajectory
import numpy as np
from pyquaternion import Quaternion
from hem.robosuite.controllers import PickPlaceController
from multiprocessing import Pool, cpu_count
import functools
import os
import pickle as pkl
import random


def save_rollout(env_type, save_dir, camera_obs=True, random_reach=False, n_per_group=1, N=0, renderer=False):
    if isinstance(N, int):
        N = [N]

    for n in N:
        if os.path.exists('{}/traj{}.pkl'.format(save_dir, n)):
            continue
        task = int((n % (16 * n_per_group)) // n_per_group)
        if random_reach:
            traj = get_random_trajectory(env_type, camera_obs, renderer)
        else:
            traj = get_expert_trajectory(env_type, camera_obs, renderer, task=task)
        pkl.dump({'traj': traj, 'env_type': env_type}, open('{}/traj{}.pkl'.format(save_dir, n), 'wb'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', default='./', help='Folder to save rollouts')
    parser.add_argument('--num_workers', default=cpu_count(), type=int, help='Number of collection workers (default=n_cores)')
    parser.add_argument('--N', default=10, type=int, help="Number of trajectories to collect")
    parser.add_argument('--per_task_group', default=1, type=int, help="Number of trajectories of same task in row")
    parser.add_argument('--env', default='SawyerPickPlaceCan', type=str, help="Environment name")
    parser.add_argument('--collect_cam', action='store_true', help="If flag then will collect camera observation")
    parser.add_argument('--renderer', action='store_true', help="If flag then will display rendering GUI")
    parser.add_argument('--random_reach', action='store_true', help="If flag then will collect random reach policy data instead of expert pickplace")
    args = parser.parse_args()
    assert args.num_workers > 0, "num_workers must be positive!"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        assert os.path.isdir(args.save_dir), "directory specified but is file and not directory!"

    if args.num_workers == 1:
        save_rollout(args.env, args.save_dir, args.collect_cam, args.random_reach, args.per_task_group, list(range(args.N)), args.renderer)
    else:
        assert not args.renderer, "can't display rendering when using multiple workers"

        with Pool(cpu_count()) as p:
            f = functools.partial(save_rollout, args.env, args.save_dir, args.collect_cam, args.random_reach, args.per_task_group)
            p.map(f, range(args.N))
