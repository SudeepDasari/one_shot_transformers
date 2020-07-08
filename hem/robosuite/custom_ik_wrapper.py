from robosuite.wrappers.ik_wrapper import IKWrapper
import numpy as np
from pyquaternion import Quaternion
import robosuite.utils.transform_utils as T


ranges = np.array([[0.44, 0.74], [-0.33, 0.5], [0.82, 1.11], [0.85, 1.08], [-1, 1], [-1, 1], [-1, 1]])
def normalize_action(action):
    norm_action = action.copy()
    for d in range(ranges.shape[0]):
        norm_action[d] = 2 * (norm_action[d] - ranges[d,0]) / (ranges[d,1] - ranges[d,0]) - 1
    return (norm_action * 128).astype(np.int32).astype(np.float32) / 128


def denormalize_action(norm_action, base_pos, base_quat):
    action = np.clip(norm_action.copy(), -1, 1)
    for d in range(ranges.shape[0]):
        action[d] = 0.5 * (action[d] + 1) * (ranges[d,1] - ranges[d,0]) + ranges[d,0]
    action[3] = action[3] - 2 if action[3] > 1 else action[3]

    cmd_quat = Quaternion(angle=action[3] * np.pi, axis=action[4:7])
    cmd_quat = np.array([cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w])
    quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
    return np.concatenate((action[:3] - base_pos, quat, action[7:]))


def project_point(point, sim, camera='frontview', frame_width=320, frame_height=320, crop=[80,0]):
    model_matrix = np.zeros((3, 4))
    model_matrix[:3, :3] = sim.data.get_camera_xmat(camera).T

    fovy = sim.model.cam_fovy[sim.model.camera_name2id(camera)]
    f = 0.5 * frame_height / np.tan(fovy * np.pi / 360)
    camera_matrix = np.array(((f, 0, frame_width / 2), (0, f, frame_height / 2), (0, 0, 1)))

    MVP_matrix = camera_matrix.dot(model_matrix)
    world_coord = np.ones((4, 1))
    world_coord[:3, 0] = point -sim.data.get_camera_xpos(camera)

    clip = MVP_matrix.dot(world_coord)
    row, col = clip[:2].reshape(-1) / clip[2]
    row, col = row, frame_height - col
    return int(max(col - crop[0], 0)), int(max(row - crop[1], 0))


def post_proc_obs(obs, env):
    if 'image' in obs:
        obs['image'] = obs['image'][80:]
    if 'depth' in obs:
        obs['depth'] = obs['depth'][80:]

    quat = Quaternion(obs['eef_quat'][[3, 0, 1, 2]])
    aa = np.concatenate(([quat.angle / np.pi], quat.axis)).astype(np.float32)
    if aa[0] < 0:
        aa[0] += 2
    obs['eef_point'] = np.array(project_point(obs['eef_pos'], env.sim))
    obs['ee_aa'] = np.concatenate((obs['eef_pos'], aa)).astype(np.float32)
    return obs


class CustomIKWrapper(IKWrapper):
    def step(self, action):
        base_pos = self.env.sim.data.site_xpos[self.eef_site_id]
        base_quat = self.env._right_hand_quat
        obs, reward, done, info = super().step(denormalize_action(action, base_pos, base_quat))
        return post_proc_obs(obs, self.env), reward, done, info

    def reset(self):
        obs = super().reset()
        return post_proc_obs(obs, self.env)

    def _get_observation(self):
        return post_proc_obs(self.env._get_observation(), self.env)
