from torch.utils.data import Dataset
import cv2
import random
import os
import torch
from hem.datasets.util import resize, crop, randomize_video
# from hem.datasets.savers.render_loader import ImageRenderWrapper
import random
import numpy as np
import io
import tqdm
from hem.datasets import get_files, load_traj
from hem.datasets.util import split_files


class AgentDemonstrations(Dataset):
    def __init__(self, root_dir=None, files=None, height=224, width=224, depth=False, normalize=True, crop=None, render_dims=None, T_context=15,
                 T_pair=0, freq=1, append_s0=False, mode='train', split=[0.9, 0.1], state_spec=None, action_spec=None, sample_sides=False,
                 color_jitter=None, rand_crop=None, rand_rotate=None, is_rad=False, rand_translate=None, rand_gray=None):
        assert mode in ['train', 'val'], "mode should be train or val!"
        assert T_context >= 2 or T_pair > 0, "Must return (s,a) pairs or context!"

        if files is None:
            all_files = get_files(root_dir)
            order = split_files(len(all_files), split, mode)
            files = [all_files[o] for o in order]

        self._files = files
        self._im_dims = (width, height)
        self._render_dims = tuple(render_dims[::-1]) if render_dims is not None else self._im_dims
        self._crop = tuple(crop) if crop is not None else (0, 0, 0, 0)
        self._depth = depth
        self._normalize = normalize
        self._T_context = T_context
        self._T_pair = T_pair
        self._freq = freq
        state_spec = tuple(state_spec) if state_spec else ('ee_aa', 'ee_vel', 'joint_pos', 'joint_vel', 'gripper_qpos', 'object_detected')
        action_spec = tuple(action_spec) if action_spec else ('action',)
        self._state_action_spec = (state_spec, action_spec)
        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_rot = rand_rotate if rand_rotate is not None else 0
        if not is_rad:
            self._rand_rot = np.radians(self._rand_rot)
        self._rand_trans = np.array(rand_translate if rand_translate is not None else [0, 0])
        self._rand_gray = rand_gray
        self._normalize = normalize
        self._append_s0 = append_s0
        self._sample_sides = sample_sides

    def __len__(self):
        return len(self._files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self._files), "invalid index!"
        return self.proc_traj(self.get_traj(index))
    
    def get_traj(self, index):
        return load_traj(self._files[index])

    def proc_traj(self, traj):
        context_frames = []
        if self._T_context:
            context_frames = self._make_context(traj)

        if self._T_pair == 0:
            return {}, context_frames
        return self._get_pairs(traj), context_frames

    def _make_context(self, traj):
        clip = lambda x : int(max(0, min(x, len(traj) - 1)))
        per_bracket = max(len(traj) / self._T_context, 1)
        def _make_frame(n):
            obs = traj.get(n)['obs']
            img = self._crop_and_resize(obs['image'])
            if self._depth:
                img = np.concatenate((img, self._crop_and_resize(obs['depth'][:,:,None])), -1)
            return img[None]

        frames = []
        for i in range(self._T_context):
            n = clip(np.random.randint(int(i * per_bracket), int((i + 1) * per_bracket)))
            if self._sample_sides and i == 0:
                n = 0
            elif self._sample_sides and i == self._T_context - 1:
                n = len(traj) - 1
            frames.append(_make_frame(n))
        frames = np.concatenate(frames, 0)
        frames = randomize_video(frames, self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize)
        return np.transpose(frames, (0, 3, 1, 2))

    def _get_pairs(self, traj, end=None):
        def _get_tensor(k, t):
            if k == 'action':
                return t['action']
            o = t['obs']
            if k == 'ee_aa' and 'ee_aa' not in o:
                ee, axis_angle = o['ee_pos'][:3], o['axis_angle']
                if axis_angle[0] < 0:
                    axis_angle[0] += 2
                o = np.concatenate((ee, axis_angle)).astype(np.float32)
            else:
                o = o[k]
            return o
        
        state_keys, action_keys = self._state_action_spec
        ret_dict = {'images': [], 'states': [], 'actions': []}
        if self._depth:
            ret_dict['depth'] = []
        if end is None:
            end = len(traj)
        start = np.random.randint(0, max(1, end - self._T_pair * self._freq))
        chosen_t = [j * self._freq + start for j in range(self._T_pair + 1)]
        if self._append_s0:
            chosen_t = [0] + chosen_t

        for j, t in enumerate(chosen_t):
            t = traj.get(t)
            if self._depth:
                depth_img = self._crop_and_resize(t['obs']['depth']).transpose((2, 0, 1))[None]
                ret_dict['depth'].append(depth_img)
            ret_dict['images'].append(self._crop_and_resize(t['obs']['image'])[None])
            state = []
            for k in state_keys:
                state.append(_get_tensor(k, t))
            ret_dict['states'].append(np.concatenate(state).astype(np.float32)[None])
            
            if j > 1 or (j==1 and not self._append_s0):
                action = []
                for k in action_keys:
                    action.append(_get_tensor(k, t))
                ret_dict['actions'].append(np.concatenate(action).astype(np.float32)[None])
        for k, v in ret_dict.items():
            ret_dict[k] = np.concatenate(v, 0).astype(np.float32)
        ret_dict['images'] = randomize_video(ret_dict['images'], self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize)
        ret_dict['images'] = np.transpose(ret_dict['images'], (0, 3, 1, 2))
        return ret_dict
    
    def _crop_and_resize(self, img, normalize=False):
        return resize(crop(img, self._crop), self._im_dims, normalize)


if __name__ == '__main__':
    import time
    import imageio
    from torch.utils.data import DataLoader
    batch_size = 10
    ag = AgentDemonstrations('./test_load', normalize=False)
    loader = DataLoader(ag, batch_size = batch_size, num_workers=8)

    start = time.time()
    timings = []
    for pairs, context in loader:
        timings.append(time.time() - start)
        print(context.shape)

        if len(timings) > 1:
            break
        start = time.time()
    print('avg ex time', sum(timings) / len(timings) / batch_size)

    out = imageio.get_writer('out1.gif')
    for t in range(context.shape[1]):
        frame = [np.transpose(fr, (1, 2, 0)) for fr in context[:, t]]
        frame = np.concatenate(frame, 1)
        out.append_data(frame.astype(np.uint8))
    out.close()
