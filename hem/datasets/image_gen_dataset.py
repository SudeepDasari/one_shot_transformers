import json
from hem.datasets.util import resize, randomize_video, split_files
from hem.datasets import load_traj
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import pickle as pkl


class GenGrip(Dataset):
    def __init__(self, root_dir, img_width=320, img_height=240, rand_crop=None, color_jitter=None, rand_gray=None, split=[0.9, 0.1], mode='train', target_downscale=4, is_traj_arr=False):
        self._root_dir = os.path.expanduser(root_dir)
        self._img_height, self._img_width = img_height, img_width
        self._is_traj_arr = is_traj_arr
        
        if self._is_traj_arr:
            files = pkl.load(open(self._root_dir, 'rb'))
            self._files = [files[i] for i in split_files(len(files), split, mode)]
        else:
            self._human_grip_times = json.load(open(os.path.join(self._root_dir, 'human_grip_timings.json'), 'r'))
            files = [k for k, v in self._human_grip_times.items() if v > 5] + ['traj{}_robot.pkl'.format(i) for i in range(len(self._human_grip_times))]
            file_inds = split_files(len(files), split, mode)
            self._files = [files[i] for i in file_inds]
        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_gray = rand_gray
        self._target_downscale = target_downscale
    
    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self._is_traj_arr:
            traj = self._files[index]
            start, mid, end = [traj[i]['obs']['image'] for i in range(3)]
        else:
            traj_name = self._files[index]
            traj = load_traj(os.path.join(self._root_dir, traj_name))

            if traj_name in self._human_grip_times:
                grip_t = self._human_grip_times[traj_name]
            else:
                obj_detected = np.concatenate([traj.get(t, False)['obs']['object_detected'] for t in range(len(traj))])
                qpos = np.concatenate([traj.get(t, False)['obs']['gripper_qpos'] for t in range(len(traj))])
                if obj_detected.any():
                    grip_t = int(np.argmax(obj_detected))
                else:
                    closed = np.isclose(qpos, 0)
                    grip_t = int(np.argmax(closed))
            
            start, mid, end = traj[np.random.randint(3)]['obs']['image'], traj[grip_t]['obs']['image'], traj[len(traj) - np.random.randint(1, 4)]['obs']['image']

        all_frs = [resize(fr, (self._img_width, self._img_height), False) for fr in (start, mid, end)]
        all_frs = randomize_video(all_frs, color_jitter=self._color_jitter, rand_gray=self._rand_gray, rand_crop=self._rand_crop, normalize=True)
        down_dim = (int(self._img_width / self._target_downscale), int(self._img_height / self._target_downscale))
        context, frame = all_frs.transpose((0, 3, 1, 2)), resize(all_frs[1], down_dim, False).transpose((2, 0, 1))
        return context, frame
