from hem.robosuite.controllers.expert_pick_place import get_expert_trajectory
import numpy as np
from pyquaternion import Quaternion
from hem.robosuite.controllers import PickPlaceController
from multiprocessing import Pool, cpu_count
import functools
import os
import pickle as pkl


def save_rollout(env_type, save_dir, camera_obs=True, N=0, renderer=False):
    if isinstance(N, int):
        N = [N]

    for n in N:
        if os.path.exists('{}/traj{}.pkl'.format(save_dir, n)):
            continue

        traj = get_expert_trajectory(env_type, camera_obs, renderer)
        pkl.dump(traj, open('{}/traj{}.pkl'.format(save_dir, n), 'wb'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', default='./', help='Folder to save rollouts')
    parser.add_argument('--num_workers', default=cpu_count(), type=int, help='Number of collection workers (default=n_cores)')
    parser.add_argument('--N', default=10, type=int, help="Number of trajectories to collect")
    parser.add_argument('--env', default='SawyerPickPlaceCan', type=str, help="Environment name")
    parser.add_argument('--no_cam', action='store_true', help="If flag then will not collect camera observation")
    parser.add_argument('--renderer', action='store_true', help="If flag then will display rendering GUI")
    args = parser.parse_args()
    assert args.num_workers > 0, "num_workers must be positive!"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    else:
        assert os.path.isdir(args.save_dir), "directory specified but is file and not directory!"

    if args.num_workers == 1:
        save_rollout(args.env, args.save_dir, not args.no_cam, list(range(args.N)), args.renderer)
    else:
        assert not args.renderer, "can't display rendering when using multiple workers"

        with Pool(cpu_count()) as p:
            n_per = int(args.N // args.num_workers)
            jobs = [range(i * n_per, (i + 1) * n_per) for i in range(args.num_workers - 1)]
            jobs.append(range((args.num_workers - 1) * n_per, args.N))
            
            f = functools.partial(save_rollout, args.env, args.save_dir, not args.no_cam)
            p.map(f, jobs)
