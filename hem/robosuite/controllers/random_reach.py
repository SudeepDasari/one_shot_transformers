from pyquaternion import Quaternion
from robosuite.controllers.sawyer_ik_controller import SawyerIKController
from robosuite.controllers.baxter_ik_controller import BaxterIKController
from robosuite.environments.baxter import BaxterEnv
from robosuite.environments.sawyer import SawyerEnv
import robosuite
import os
import numpy as np
from hem.robosuite import get_env
from hem.datasets import Trajectory
import pybullet as p


def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)

    if norm_delta < max_step:
        return np.zeros_like(delta)
    return delta / norm_delta * max_step


class RandomController:
    def __init__(self, env):
        assert env.single_object_mode == 2, "only supports single object environments at this point!"
        self._env = env
        self.reset()
        
    def _calculate_quat(self, angle):
        if isinstance(self._env, SawyerEnv):
            new_rot = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
            return Quaternion(matrix=self._base_rot.dot(new_rot))
        return self._base_quat
    
    def reset(self):
        # TODO this line violates abstraction barriers but so does the reference implementation in robosuite
        self._jpos_getter = lambda : np.array(self._env._joint_positions)
        bullet_path = os.path.join(robosuite.models.assets_root, "bullet_data")

        if isinstance(self._env, SawyerEnv):
            self._ik = SawyerIKController(bullet_path, self._jpos_getter)
            self._obs_name = 'eef_pos'
            self._default_speed = 0.010
            self._final_thresh = 1e-2
            self._range = np.array([[0.43730527, -0.32466202,  0.83998252], [0.72514142, 0.50403689, 1.05429141]])
        elif isinstance(self._env, BaxterEnv):
            self._ik = BaxterIKController(bullet_path, self._jpos_getter)
            self._obs_name = 'right_eef_pos'
            self._default_speed = 0.002
            self._final_thresh = 6e-2
            self._range = np.array([[0.45848022, -0.42518796, 0.8359104], [0.87102066, 0.36041984, 1.05087887]])
        else:
            raise NotImplementedError

        self._t = 0
        self._base_rot = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        self._base_quat = Quaternion(matrix=self._base_rot)
        self._gripper = -1
        self._p = 0.01
    
    def _get_velocities(self, delta_pos, quat, max_step=None):
        if max_step is None:
            max_step = self._default_speed
        
        if isinstance(self._env, BaxterEnv):
            right_delta = {'dpos': _clip_delta(delta_pos, max_step), 'rotation': quat.transformation_matrix[:3,:3]}
            left_delta = {'dpos': np.zeros(3), 'rotation': self._base_rot}
            velocities = self._ik.get_control(right_delta, left_delta)
            return velocities[:7]
        return self._ik.get_control(_clip_delta(delta_pos, max_step), quat.transformation_matrix[:3,:3])
    
    def act(self, obs):
        if self._t == 0:
            self._target = np.random.uniform(self._range[0], self._range[1])
            y = -(self._target[1] - obs[self._obs_name][1])
            x = self._target[0] - obs[self._obs_name][0]
            self._target_quat = self._calculate_quat(np.arctan2(y, x))
        
        if np.random.uniform() < self._p:
            self._gripper *= -1

        quat_t = Quaternion.slerp(self._base_quat, self._target_quat, min(1, float(self._t) / 50))
        velocities = self._get_velocities(self._target - obs[self._obs_name], quat_t)
        action = np.concatenate((velocities, [self._gripper]))
        self._t += 1
        return action

    def disconnect(self):
        p.disconnect()


def get_random_trajectory(env_type, camera_obs=True, renderer=False, horizon=200):
    success = False
    while not success:
        np.random.seed()
        env = get_env(env_type)(has_renderer=renderer, reward_shaping=False, use_camera_obs=camera_obs, horizon=horizon)
        obs = env.reset()
        mj_state = env.sim.get_state().flatten()
        sim_xml = env.model.get_xml()
        traj = Trajectory(sim_xml)

        env.reset_from_xml_string(sim_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(mj_state)
        env.sim.forward()
        controller = RandomController(env)

        traj.append(obs, raw_state=mj_state)
        for _ in range(env.horizon):
            action = controller.act(obs)
            obs, reward, done, info = env.step(action)
            mj_state = env.sim.get_state().flatten()
            traj.append(obs, reward, done, info, action, mj_state)
            
            if reward or done:
                success = True
                break
            if renderer:
                env.render()
    
    if renderer:
        env.close()
    
    controller.disconnect()
    return traj
