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
    action = norm_action.copy()
    for d in range(ranges.shape[0]):
        action[d] = 0.5 * (action[d] + 1) * (ranges[d,1] - ranges[d,0]) + ranges[d,0]
    action[3] = action[3] - 2 if action[3] > 1 else action[3]

    cmd_quat = Quaternion(angle=action[3] * np.pi, axis=action[4:7])
    cmd_quat = np.array([cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w])
    quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
    return np.concatenate((action[:3] - base_pos, quat, action[7:]))


class CustomIKWrapper(IKWrapper):
    def step(self, action):
        base_pos = self.env.sim.data.site_xpos[self.eef_site_id]
        base_quat = self.env._right_hand_quat
        return super().step(denormalize_action(action, base_pos, base_quat))
