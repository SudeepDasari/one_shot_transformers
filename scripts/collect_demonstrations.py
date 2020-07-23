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


def save_rollout(env_type, save_dir, n_tasks, env_seed=False, force=False, camera_obs=True, seeds=None, n_per_group=1, N=0, renderer=False):
    if isinstance(N, int):
        N = [N]

    for n in N:
        if os.path.exists('{}/traj{}.pkl'.format(save_dir, n)):
            continue
        task = int((n % (n_tasks * n_per_group)) // n_per_group)
        seed = None if seeds is None else seeds[n]
        env_seed = seeds[n - n % n_tasks] if seeds is not None and env_seed else None
        traj = get_expert_trajectory(env_type, camera_obs, renderer, task=task, seed=seed, force_success=force, env_seed=env_seed)
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
    parser.add_argument('--random_seed', action='store_true', help="If flag then will collect data from random envs")
    parser.add_argument('--n_env', default=None, type=int, help="Number of environments to collect from")
    parser.add_argument('--n_tasks', default=16, type=int, help="Number of tasks in environment")
    parser.add_argument('--give_env_seed', action='store_true', help="Maintain seperate consistent environment sampling seed (for multi obj envs)")
    parser.add_argument('--force', action='store_true', help="Use this flag for teacher demos where 'good' control signals don't matter")
    args = parser.parse_args()
    assert args.num_workers > 0, "num_workers must be positive!"

    if args.random_seed:
        assert args.n_env is None
        seeds = [None for _ in range(args.N)]
    elif args.n_env:
        envs, rng = [263237945 + i for i in range(args.n_env)], random.Random(385008283)
        seeds = [int(rng.choice(envs)) for _ in range(args.N)]
    else:
        n_per_group = args.per_task_group
        seeds = [263237945 + int(n // (args.n_tasks * n_per_group)) * n_per_group + n % n_per_group for n in range(args.N)]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        assert os.path.isdir(args.save_dir), "directory specified but is file and not directory!"

    if args.num_workers == 1:
        save_rollout(args.env, args.save_dir, args.n_tasks, args.give_env_seed, args.force, args.collect_cam, seeds, args.per_task_group, list(range(args.N)), args.renderer)
    else:
        assert not args.renderer, "can't display rendering when using multiple workers"

        with Pool(cpu_count()) as p:
            f = functools.partial(save_rollout, args.env, args.save_dir, args.n_tasks, args.give_env_seed, args.force, args.collect_cam, seeds, args.per_task_group)
            p.map(f, range(args.N))
