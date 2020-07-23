from torch.utils.data import Dataset
import cv2
import random
import os
import torch
from hem.datasets.util import resize, crop, randomize_video
import random
import numpy as np
import io
import tqdm
from hem.datasets import get_files, load_traj
from hem.datasets.util import split_files
import pickle as pkl


class AgentDemonstrations(Dataset):
    def __init__(self, root_dir=None, files=None, height=224, width=224, depth=False, normalize=True, crop=None, randomize_vid_frames=False, T_context=15, extra_samp_bound=0,
                 T_pair=0, freq=1, append_s0=False, mode='train', split=[0.9, 0.1], state_spec=None, action_spec=None, sample_sides=False, min_frame=0, cache=False, random_targets=False,
                 color_jitter=None, rand_crop=None, rand_rotate=None, is_rad=False, rand_translate=None, rand_gray=None, rep_buffer=0, target_vid=False, reduce_bits=False, aux_pose=False):
        assert mode in ['train', 'val'], "mode should be train or val!"
        assert T_context >= 2 or T_pair > 0, "Must return (s,a) pairs or context!"

        if files is None and rep_buffer:
            all_files = []
            for f in range(rep_buffer):
                all_files.extend(pkl.load(open(os.path.expanduser(root_dir.format(f)), 'rb')))
            order = split_files(len(all_files), split, mode)
            files = [all_files[o] for o in order]
        elif files is None:
            all_files = get_files(root_dir)
            order = split_files(len(all_files), split, mode)
            files = [all_files[o] for o in order]

        self._trajs = files
        if cache:
            for i in tqdm.tqdm(range(len(self._trajs))):
                if isinstance(self._trajs[i], str):
                    self._trajs[i] = load_traj(self._trajs[i])

        self._im_dims = (width, height)
        self._randomize_vid_frames = randomize_vid_frames
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
        self._target_vid = target_vid
        self._reduce_bits = reduce_bits
        self._min_frame = min_frame
        self._extra_samp_bound = extra_samp_bound
        self._random_targets = random_targets
        self._aux_pose = aux_pose

    def __len__(self):
        return len(self._trajs)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self._trajs), "invalid index!"
        return self.proc_traj(self.get_traj(index))
    
    def get_traj(self, index):
        if isinstance(self._trajs[index], str):
            return load_traj(self._trajs[index])
        return self._trajs[index]

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
                n = self._min_frame
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
            elif k == 'grip_action':
                return [t['action'][-1]]

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
        has_eef_point = 'eef_point' in traj.get(0, False)['obs']
        if has_eef_point:
            ret_dict['points'] = []

        end = len(traj) if end is None else end
        start = np.random.randint(0, max(1, end - self._T_pair * self._freq))
        if np.random.uniform() < self._extra_samp_bound:
            start = 0 if np.random.uniform() < 0.5 else max(1, end - self._T_pair * self._freq) - 1
        chosen_t = [j * self._freq + start for j in range(self._T_pair + 1)]
        if self._append_s0:
            chosen_t = [0] + chosen_t

        for j, t in enumerate(chosen_t):
            t = traj.get(t)
            if self._depth:
                depth_img = self._crop_and_resize(t['obs']['depth']).transpose((2, 0, 1))[None]
                ret_dict['depth'].append(depth_img)
            image = t['obs']['image']
            ret_dict['images'].append(self._crop_and_resize(image)[None])
            
            if has_eef_point:
                ret_dict['points'].append(np.array(self._adjust_points(t['obs']['eef_point'], image.shape[:2]))[None])
    
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
        if self._target_vid:
            ret_dict['target_images'] = randomize_video(ret_dict['images'].copy(), normalize=False).transpose((0, 3, 1, 2))
        if self._random_targets:
            ret_dict['transformed'] = [randomize_video([f], self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize) for f in ret_dict['images']]
            ret_dict['transformed'] = np.concatenate(ret_dict['transformed'], 0).transpose(0, 3, 1, 2)

        if self._randomize_vid_frames:
            ret_dict['images'] = [randomize_video([f], self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize) for f in ret_dict['images']]
            ret_dict['images'] = np.concatenate(ret_dict['images'], 0)
        else:
            ret_dict['images'] = randomize_video(ret_dict['images'], self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize)
        ret_dict['images'] = np.transpose(ret_dict['images'], (0, 3, 1, 2))

        if self._aux_pose:
            grip_close = np.array([traj.get(i, False)['action'][-1] > 0 for i in range(1, len(traj))])
            grip_t = np.argmax(grip_close)
            drop_t = len(traj) - 1 - np.argmax(np.logical_not(grip_close)[::-1])
            aux_pose = [traj.get(t, False)['obs']['ee_aa'][:3] for t in (grip_t, drop_t)]
            ret_dict['aux_pose'] = np.concatenate(aux_pose).astype(np.float32)
        return ret_dict
    
    def _crop_and_resize(self, img, normalize=False):
        return resize(crop(img, self._crop), self._im_dims, normalize, self._reduce_bits)
    
    def _adjust_points(self, points, frame_dims):
        h = np.clip(points[0] - self._crop[0], 0, frame_dims[0] - self._crop[1])
        w = np.clip(points[1] - self._crop[2], 0, frame_dims[1] - self._crop[3])
        h = float(h) / (frame_dims[0] - self._crop[0] - self._crop[1]) * self._im_dims[1]
        w = float(w) / (frame_dims[1] - self._crop[2] - self._crop[3]) * self._im_dims[0]
        return tuple([int(min(x, d - 1)) for x, d in zip([h, w], self._im_dims[::-1])])


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
