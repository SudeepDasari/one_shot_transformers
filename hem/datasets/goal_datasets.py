from hem.datasets import load_traj, get_files
from torch.utils.data import Dataset
from hem.datasets.util import resize, randomize_video, split_files, select_random_frames
from hem.datasets.util import crop as crop_frame
import torch
import tqdm
import os
import numpy as np


class ContrastiveGoals(Dataset):
    def __init__(self, root_dir, DEMOS_PER_TASK=8, TASK_PER_SET=5, splits=[0.9, 0.1], mode='train', T=3, height=240, width=320, crop=(0,0,0,0), rand_crop=None, color_jitter=None, n_neg=16):
        assert n_neg <= DEMOS_PER_TASK * (TASK_PER_SET - 1)
        files = get_files(root_dir)
        tasks = split_files(int(len(files) / (DEMOS_PER_TASK * TASK_PER_SET)), splits, mode)
        self._DEMOS_PER_TASK = DEMOS_PER_TASK
        self._TASK_PER_SET = TASK_PER_SET
        self._n_neg, self._T = n_neg, T
        self._rand_crop = rand_crop
        self._color_jitter = color_jitter
        self._cache, self._index_to_task = {}, []
        for t in tqdm.tqdm(tasks):
            for i in range(t * DEMOS_PER_TASK * TASK_PER_SET, (t + 1) * DEMOS_PER_TASK * TASK_PER_SET):
                self._index_to_task.append(i)
                traj = load_traj(os.path.join(os.path.expanduser(root_dir), 'traj{}.pkl'.format(i)))
                frames = select_random_frames(traj, T, sample_sides=True)
                self._cache[i] = [resize(crop_frame(fr, crop), (width, height), normalize=False).astype(np.uint8) for fr in frames]

    def __len__(self):
        return len(self._index_to_task)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        index = self._index_to_task[index]
        obj_set_id = int(index / (self._DEMOS_PER_TASK * self._TASK_PER_SET))
        task_id = int((index % (self._DEMOS_PER_TASK * self._TASK_PER_SET)) / self._DEMOS_PER_TASK)
        video_id = index % self._DEMOS_PER_TASK

        other_tasks = [t for t in range(self._TASK_PER_SET) if t != task_id]
        negative_indices = []
        while len(negative_indices) < self._n_neg:
            samp_task, samp_vid = np.random.choice(other_tasks), np.random.choice(self._DEMOS_PER_TASK)
            samp_index = (obj_set_id * self._TASK_PER_SET + samp_task) * self._DEMOS_PER_TASK + samp_vid
            if samp_index not in negative_indices:
                negative_indices.append(samp_index)

        other_positive = np.random.choice([t for t in range(self._DEMOS_PER_TASK) if t != video_id])
        other_positive = (obj_set_id * self._TASK_PER_SET + task_id) * self._DEMOS_PER_TASK + other_positive

        frames = self._cache[index] + self._cache[other_positive][-1:] + [self._cache[n][-1] for n in negative_indices]
        query = randomize_video(self._cache[index], rand_crop=self._rand_crop, color_jitter=self._color_jitter, normalize=True)
        positive = randomize_video(self._cache[other_positive], rand_crop=self._rand_crop, color_jitter=self._color_jitter, normalize=True)
        negatives = [randomize_video(self._cache[n], rand_crop=self._rand_crop, color_jitter=self._color_jitter, normalize=True) for n in negative_indices]
        negatives = np.concatenate([n[None] for n in negatives], 0)
        return query.transpose((0, 3, 1, 2)).astype(np.float32), positive.transpose((0, 3, 1, 2)).astype(np.float32), negatives.transpose((0, 1, 4, 2, 3)).astype(np.float32)
