import unittest
from hem.robosuite import get_env
import numpy as np
from pyquaternion import Quaternion
from hem.robosuite.controllers import PickPlaceController
from multiprocessing import Pool, cpu_count
import functools


def expert_rollout(N, env_type, render):
    env = get_env(env_type)(has_renderer=render, reward_shaping=False, use_camera_obs=False)
    controller = PickPlaceController(env)

    success = 0
    for _ in range(N):
        obs = env.reset()
        controller.reset()

        if render:
            env.render()
        
        for t in range(env.horizon):
            obs, reward, done, info = env.step(controller.act(obs))
            if render:
                env.render()
            if reward:
                success += 1
                break
        if render:
            env.close()
    return success


class TestPickPlaceControllers(unittest.TestCase):
    N = 150

    def test_baxter_can(self):
        n_per = int(np.ceil(self.N / cpu_count()))
        N = n_per * cpu_count()

        with Pool(cpu_count()) as p:
            f = functools.partial(expert_rollout, env_type='BaxterPickPlaceCan', render=False)
            success = sum(p.map(f, [n_per for _ in range(cpu_count())]))
        self.assertEqual(success / N, 1)
    
    def test_sawyer_can(self):
        n_per = int(np.ceil(self.N / cpu_count()))
        N = n_per * cpu_count()

        with Pool(cpu_count()) as p:
            f = functools.partial(expert_rollout, env_type='SawyerPickPlaceCan', render=False)
            success = sum(p.map(f, [n_per for _ in range(cpu_count())]))
        self.assertEqual(success / N, 1)
    
    def test_baxter_milk(self):
        n_per = int(np.ceil(self.N / cpu_count()))
        N = n_per * cpu_count()

        with Pool(cpu_count()) as p:
            f = functools.partial(expert_rollout, env_type='BaxterPickPlaceMilk', render=False)
            success = sum(p.map(f, [n_per for _ in range(cpu_count())]))
        self.assertEqual(success / N, 1)
    
    def test_sawyer_milk(self):
        n_per = int(np.ceil(self.N / cpu_count()))
        N = n_per * cpu_count()

        with Pool(cpu_count()) as p:
            f = functools.partial(expert_rollout, env_type='SawyerPickPlaceMilk', render=False)
            success = sum(p.map(f, [n_per for _ in range(cpu_count())]))
        self.assertEqual(success / N, 1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_test', action='store_true', help='Skip test cases')
    parser.add_argument('--render', action='store_true', help='Render to openGL viewer')
    parser.add_argument('--N', default=10, type=int, help='Number of trails')
    parser.add_argument('--env', default='SawyerPickPlaceCan', type=str, help='Environment name')
    args = parser.parse_args()
    
    if not args.no_test:
        unittest.main()
        exit(0)

    if args.render:
        success = expert_rollout(args.N, args.env, True)
    else:
        with Pool(cpu_count()) as p:
            f = functools.partial(expert_rollout, env_type=args.env, render=False)
            n_per = int(np.ceil(args.N / cpu_count()))
            args.N = n_per * cpu_count()
            success = sum(p.map(f, [n_per for _ in range(cpu_count())]))
    print('Success rate', success / float(args.N))
