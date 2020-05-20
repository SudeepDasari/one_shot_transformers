from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random
from hem.datasets import get_files, load_traj
from hem.datasets.util import resize, crop, randomize_video
import cv2
import tqdm
import multiprocessing
from hem.datasets.util import split_files
import json


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
        traj = load_traj(self._file)
        self.add(index, traj[index]['obs']['image'])
        return traj[index]


def _build_cache(traj_file):
    traj = load_traj(traj_file)
    cache = _CachedTraj(traj_file, len(traj))
    for i in range(max(1, int(len(traj) // 3))):
        cache.add(i, traj[i]['obs']['image'])
    for i in range(int(2 * len(traj) / 3.0), len(traj)):
        cache.add(i, traj[i]['obs']['image'])
    return cache


class PairedFrameDataset(Dataset):
    def __init__(self, root_dir, mode='train', split=[0.9, 0.1], color_jitter=None, rand_crop=None, rand_rotate=None, is_rad=False, rand_translate=None, rand_gray=None, normalize=True, crop=None, height=224, width=224, cache=None, teacher_first=False):
        root_dir = os.path.expanduser(root_dir)
        map_file = json.load(open(os.path.join(root_dir, 'mappings_1_to_1.json'), 'r'))
        teacher_files = sorted(list(map_file.keys()))
        agent_files = [os.path.join(root_dir, map_file[k]) for k in teacher_files]
        teacher_files = [os.path.join(root_dir, teacher_fname) for teacher_fname in teacher_files]
        assert len(agent_files) == len(teacher_files), "lengths don't match!"

        order = split_files(len(agent_files), split, mode)
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
        self._teacher_first = teacher_first

        self._cache = None
        if cache:
            self._cache = {}
            jobs = self._agent_files + self._teacher_files
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                cache_objs = list(tqdm.tqdm(p.imap(_build_cache, jobs), total=len(jobs)))
            for j, c in zip(jobs, cache_objs):
                self._cache[j] = c

    def __len__(self):
        return len(self._agent_files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        agent, teacher = self._agent_files[index], self._teacher_files[index]
        chosen_slice = np.random.randint(self._slices)
        agent_fr, teacher_fr = self._format_img(self._slice(agent, chosen_slice)), self._format_img(self._slice(teacher, chosen_slice, True))
        if np.random.uniform() < 0.5 and not self._teacher_first:
            return agent_fr, teacher_fr
        return teacher_fr, agent_fr
    
    def _load(self, traj_file):
        if self._cache is None:
            return load_traj(traj_file)
        if traj_file in self._cache:
            return self._cache[traj_file]
        
        traj = load_traj(traj_file)
        cached = _CachedTraj(traj_file, len(traj))
        for i in range(int(len(traj) // 3)):
            cached.add(i, traj[i]['obs']['image'])
        for i in range(max(len(traj) - int(len(traj) // 3), 0), len(traj)):
            cached.add(i, traj[i]['obs']['image'])
        self._cache[traj_file] = cached
        return traj

    def _slice(self, traj, chosen_slice, is_teacher=False):
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


class SyncedFramesDataset(PairedFrameDataset):
    TEACHER_TIME_MARGIN = 6
    AGENT_STATE_MARGIN = 0.1
    def __init__(self, root_dir, **kwargs):
        kwargs.pop('cache', None)
        root_dir = os.path.expanduser(root_dir)
        with open(os.path.join(root_dir, 'human_grip_timings.json'), 'r') as f:
            self._timing_map = json.load(f)
            for k in list(self._timing_map.keys()):
                self._timing_map[os.path.join(root_dir, k)] = self._timing_map.pop(k)
        super().__init__(root_dir, **kwargs)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"
        agent, teacher = load_traj(self._agent_files[index]), load_traj(self._teacher_files[index])

        # get grip timings for sampling
        obj_detected = np.concatenate([agent.get(t, False)['obs']['object_detected'] for t in range(len(agent))])
        qpos = np.concatenate([agent.get(t, False)['obs']['gripper_qpos'] for t in range(len(agent))])
        if obj_detected.any():
            agent_grip_t = int(np.argmax(obj_detected))
        else:
            closed = np.isclose(qpos, 0)
            agent_grip_t = int(np.argmax(closed))
        grip_loc = agent.get(agent_grip_t, False)['obs']['ee_pos'][:3]
        teacher_grip_t = self._timing_map[self._teacher_files[index]]

        phase = ['START', 'GRIP', 'DROP']
        if teacher_grip_t < self.TEACHER_TIME_MARGIN:
            phase = phase[1:]
        phase = phase[np.random.randint(len(phase))]

        if phase == 'START':
            teacher_t = np.random.randint(teacher_grip_t - self.TEACHER_TIME_MARGIN + 1)
            agent_t = []
            for t_prime in range(agent_grip_t):
                obs_t_prime = agent.get(t_prime, False)['obs']
                delta = np.linalg.norm(obs_t_prime['ee_pos'][:3] - grip_loc)
                if  delta > 1.25 * self.AGENT_STATE_MARGIN:
                    agent_t.append(t_prime)
            agent_t = int(np.random.choice(agent_t)) if len(agent_t) else 0
            teacher_hard_neg = np.random.randint(teacher_grip_t - int(0.4 * self.TEACHER_TIME_MARGIN), len(teacher))
        elif phase == 'GRIP':
            teacher_t, agent_t = teacher_grip_t, agent_grip_t
            if teacher_grip_t < self.TEACHER_TIME_MARGIN or np.random.uniform() < 0.5:
                teacher_hard_neg = np.random.randint(min(len(teacher) - 1, teacher_grip_t + int(1.5 * self.TEACHER_TIME_MARGIN)), len(teacher))
            else:
                teacher_hard_neg = np.random.randint(max(1, teacher_grip_t - int(1.5 * self.TEACHER_TIME_MARGIN) + 1))
        else:
            teacher_t = np.random.randint(len(teacher) - self.TEACHER_TIME_MARGIN, len(teacher))
            agent_t = []
            for t_prime in range(agent_grip_t, len(agent)):
                obs_t_prime = agent.get(t_prime, False)['obs']
                delta = np.linalg.norm(obs_t_prime['ee_pos'][:3] - grip_loc)
                if  delta > 1.25 * self.AGENT_STATE_MARGIN:
                    agent_t.append(t_prime)
            agent_t = int(np.random.choice(agent_t)) if len(agent_t) else len(agent) - 1
            teacher_hard_neg = np.random.randint(min(teacher_grip_t + int(0.4 * self.TEACHER_TIME_MARGIN), len(teacher)))

        agent_t = agent[agent_t]['obs']
        agent_hard_neg = []
        for t_prime in range(len(agent)):
            obs_t_prime = agent.get(t_prime, False)['obs']
            delta = np.linalg.norm(obs_t_prime['ee_pos'][:3] - agent_t['ee_pos'][:3])
            if  delta > self.AGENT_STATE_MARGIN or (obs_t_prime['object_detected'] != agent_t['object_detected'] and delta > 0.5 * self.AGENT_STATE_MARGIN):
                agent_hard_neg.append(t_prime)
        agent_hard_neg = int(agent_hard_neg[np.random.randint(len(agent_hard_neg))])

        K, Q = teacher[teacher_t]['obs']['image'], agent_t['image']
        N = agent[agent_hard_neg]['obs']['image'] if np.random.uniform() < 0.5 else teacher[teacher_hard_neg]['obs']['image']
        if np.random.uniform() < 0.5:
            K, Q = Q, K
        K = self._format_img(K)
        Q, N = self._format_vid([Q, N])
        return K, Q, N

    def _format_vid(self, vid):
        vid = [resize(crop(img, self._crop), self._im_dims, False) for img in vid]
        return np.transpose(randomize_video(vid, self._color_jitter, self._rand_gray, self._rand_crop, self._rand_rot, self._rand_trans, self._normalize), (0, 3, 1, 2))
