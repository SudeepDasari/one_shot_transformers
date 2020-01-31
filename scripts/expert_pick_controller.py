from hem.robosuite import get_env
from robosuite.controllers.sawyer_ik_controller import SawyerIKController
import robosuite
import os
import numpy as np
from pyquaternion import Quaternion


def get_delta(obs):
    delta = obs['Can0_pos'] - obs['eef_pos']
    norm_delta = np.linalg.norm(delta)

    if norm_delta < 0.01:
        return np.zeros_like(delta)
    return delta / norm_delta * 0.01


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--T', default=200, type=int)
    args = parser.parse_args()
    
    main_env = get_env('SawyerPickPlaceCan')(has_renderer=args.render, reward_shaping=True, use_camera_obs=False)
    jpos_getter = lambda : np.array(main_env._joint_positions)
    sawyer_ik = SawyerIKController(os.path.join(robosuite.models.assets_root, "bullet_data"), jpos_getter)
    
    obs = main_env.reset()
    if args.render:
        main_env.render()
    
    base_rot = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    angle = np.arctan2(-(obs['Can0_pos'][1] - obs['eef_pos'][1]), obs['Can0_pos'][0] - obs['eef_pos'][0])
    new_rot = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
    quat = Quaternion(matrix=base_rot)
    new_quat = Quaternion(matrix=base_rot.dot(new_rot))

    print(obs['eef_pos'])
    print(obs['Can0_pos'] - obs['eef_pos'])

    velocities = sawyer_ik.get_control(get_delta(obs), np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]))
    for t in range(args.T):
        target_quat = Quaternion.slerp(quat, new_quat, min(1, float(t) / 50))
        obs = main_env.step(np.concatenate((velocities, [-1])))[0]
        if args.render:
            main_env.render()
        velocities = sawyer_ik.get_control(get_delta(obs), target_quat.transformation_matrix[:3,:3])
    print(jpos_getter())