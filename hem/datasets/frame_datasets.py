from torch.utils.data import Dataset
from .agent_dataset import SHUFFLE_RNG
import torch
import os
import numpy as np
import glob
import random
from hem.datasets.util import resize, crop, randomize_video
import pickle as pkl
import cv2
import tqdm


class _CachedTraj:
    def __init__(self, traj_file, traj_len):
        self._img_cache = {}
        self._file = traj_file
        self._len = traj_len

    def __len__(self):
        return self._len

    def add(self, index, img):
        assert 0 <= index < len(self), "index out of bounds!"
        status, buf = cv2.imencode('.jpg', img)
        assert status, "compression failed"
        self._img_cache[index] = buf
    
    def __getitem__(self, index):
        assert 0 <= index < len(self), "index out of bounds!"

        if index in self._img_cache:
            return {'obs': {'image':cv2.imdecode(self._img_cache[index], cv2.IMREAD_COLOR)}}
        traj = pkl.load(open(self._file, 'rb'))['traj']
        self.add(index, traj[index]['obs']['image'])
        return traj[index]

    
class PairedFrameDataset(Dataset):
    def __init__(self, root_dir, mode='train', split=[0.9, 0.1], color_jitter=None, rand_crop=None, rand_rotate=None, is_rad=False, rand_translate=None, rand_gray=None, normalize=True, crop=None, height=224, width=224, cache=None):
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

        self._cache = None
        if cache:
            self._cache = {}
            print('caching agents...')
            for a in tqdm.tqdm(self._agent_files):
                traj = pkl.load(open(a, 'rb'))['traj']
                cached = _CachedTraj(a, len(traj))
                for i in range(5):
                    cached.add(i, traj[i]['obs']['image'])
                for i in range(max(len(traj) - 5, 0), len(traj)):
                    cached.add(i, traj[i]['obs']['image'])
                self._cache[a] = cached
            print('caching teacher...')
            for t in tqdm.tqdm(self._teacher_files):
                traj = pkl.load(open(t, 'rb'))['traj']
                cached = _CachedTraj(t, len(traj))
                for i in range(max(int(len(traj) // 3), 5)):
                    cached.add(i, traj[i]['obs']['image'])
                for i in range(max(len(traj) - max(int(len(traj) // 12), 5), 0), len(traj)):
                    cached.add(i, traj[i]['obs']['image'])
                self._cache[t] = cached

    def __len__(self):
        return len(self._agent_files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        agent, teacher = self._agent_files[index], self._teacher_files[index]
        chosen_slice = np.random.randint(self._slices)
        agent_fr, teacher_fr = self._format_img(self._slice(agent, chosen_slice)), self._format_img(self._slice(teacher, chosen_slice, True))
        if np.random.uniform() < 0.5:
            return agent_fr, teacher_fr
        return teacher_fr, agent_fr
    
    def _load(self, traj_file):
        if self._cache is None:
            return pkl.load(open(traj_file, 'rb'))['traj']
        if traj_file in self._cache:
            return self._cache[traj_file]
        
        traj = pkl.load(open(traj_file, 'rb'))['traj']
        self._cache[traj_file] = _CachedTraj(traj_file, len(traj))
        return traj

    def _slice(self, traj, chosen_slice, is_teacher=False):
        tf = traj
        traj = self._load(traj)
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
        
        chosen_slice = np.random.randint(self._slices)
        img = self._slice(traj, chosen_slice, is_teacher)
        return self._format_img(img.copy()), self._format_img(img.copy())
