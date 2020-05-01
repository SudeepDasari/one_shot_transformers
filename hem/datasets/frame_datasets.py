from torch.utils.data import Dataset
from .agent_dataset import SHUFFLE_RNG
import torch
import os
import numpy as np
import glob
import random
from hem.datasets.util import resize, crop, randomize_video
import pickle as pkl


class PairedFrameDataset(Dataset):
    def __init__(self, root_dir, mode='train', split=[0.9, 0.1], color_jitter=None, rand_crop=None, rand_rotate=None, is_rad=False, rand_translate=None, rand_gray=None, normalize=True, crop=None, height=224, width=224):
        assert all([0 <= s <=1 for s in split]) and sum(split)  == 1, "split not valid!"
        agent_files, teacher_files = sorted(glob.glob(os.path.join(root_dir, 'traj*_robot.pkl'))), sorted(glob.glob(os.path.join(root_dir, 'traj*_human.pkl')))
        assert len(agent_files) == len(teacher_files), "lengths don't match!"

        order = [i for i in range(len(agent_files))]
        pivot = int(len(order) * split[0])
        if mode == 'train':
            order = order[:pivot]
        else:
            order = order[pivot:]
        random.Random(SHUFFLE_RNG).shuffle(order)
        self._agent_files = [agent_files[o] for o in order]
        self._teacher_files = [teacher_files[o] for o in order]

        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_rot = rand_rotate if rand_rotate is not None else 0
        if not is_rad:
            self._rand_rot = np.radians(self._rand_rot)
        self._rand_trans = np.array(rand_translate if rand_translate is not None else [0, 0])
        self._rand_gray = rand_gray
        self._normalize = normalize
        self._slices = 2
        self._crop = tuple(crop) if crop is not None else (0, 0, 0, 0)
        self._im_dims = (width, height)

    def __len__(self):
        return len(self._agent_files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        agent = pkl.load(open(self._agent_files[index], 'rb'))['traj']
        teacher = pkl.load(open(self._teacher_files[index], 'rb'))['traj']
        chosen_slice = np.random.randint(self._slices)
        agent_fr, teacher_fr = self._format_img(self._slice(agent, chosen_slice)), self._format_img(self._slice(teacher, chosen_slice, True))
        if np.random.uniform() < 0.5:
            return agent_fr, teacher_fr
        return teacher_fr, agent_fr

    def _slice(self, traj, chosen_slice, is_teacher=False):
        delta = max(int(len(traj) // 3), 1)
        if chosen_slice == 1 and not is_teacher:
            delta =  max(int(len(traj) // 12), 1)
        if is_teacher:
            delta = min(delta, 5)
        
        if chosen_slice == 0:
            fr = np.random.randint(delta)
        else:
            fr = np.random.randint(max(0, len(traj) - delta), len(traj))
        return traj[min(len(traj) - 1, fr)]['obs']['image']

    def _format_img(self, img):
        resized = resize(crop(img, self._crop), self._im_dims, False)
        return np.transpose(randomize_video([resized], self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize)[0], (2, 0, 1))


class UnpairedFrameDataset(PairedFrameDataset):
    def __len__(self):
        return len(self._agent_files) * 2
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        is_teacher = index >= len(self._agent_files)
        if not is_teacher:
            traj = self._agent_files[index]
        else:
            traj = self._teacher_files[index - len(self._agent_files)]
        
        traj = pkl.load(open(traj, 'rb'))['traj']
        chosen_slice = np.random.randint(self._slices)
        img = self._slice(traj, chosen_slice, is_teacher)
        return self._format_img(img.copy()), self._format_img(img.copy())
