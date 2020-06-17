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
import pickle as pkl


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
        if phase == 'GRIP' and np.random.uniform() < 0.75:
            for i in range(5, 70):
                t_prime = agent_grip_t - i
                obs_t_prime = agent.get(t_prime, False)['obs']
                z_delta = obs_t_prime['ee_pos'][3] - agent_t['ee_pos'][3]
                if z_delta < 0.2 and z_delta > 0.05:
                    agent_hard_neg.append(t_prime)
        if not len(agent_hard_neg):
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


class AuxContrastiveDataset(Dataset):
    def __init__(self, root_dir, replay_buf=None, mode='train', split=[0.9, 0.1], target_downscale=4, img_width=320, img_height=240, rand_crop=None, color_jitter=None, rand_gray=None, g_sep=5, normalize=True, crop=None):
        root_dir = os.path.expanduser(root_dir)
        if replay_buf is not None:
            trajs = []
            for t in range(replay_buf):
                trajs.extend(pkl.load(open('{}_{}.pkl'.format(root_dir, t), 'rb')))
        else:
            trajs = get_files(root_dir)
        self._trajs = [trajs[i] for i in split_files(len(trajs), split, mode)]

        self._ctxt_dims, self._target_dims = (img_width, img_height), (int(img_width / target_downscale), int(img_height / target_downscale))
        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_gray = rand_gray
        self._g_sep = g_sep
        self._normalize = normalize
        self._crop = tuple(crop) if crop is not None else (0, 0, 0, 0)

    def __len__(self):
        return len(self._trajs)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        traj = self._trajs[index]
        traj = load_traj(traj) if isinstance(traj, str) else traj

        is_agent = 'agent' in traj.config_str
        obj_key = [k for k in traj[0]['obs'].keys() if all([b not in k for b in ('joint', 'eef', 'gripper')]) and 'pos' in k][0]
        obj_pos = np.concatenate([traj[i]['obs'][obj_key][None] for i in range(len(traj))], 0)
        t_q = int(np.random.randint(len(traj)))
        valid_positives = [t for t in range(len(traj)) if np.linalg.norm(obj_pos[t] - obj_pos[t_q]) <= 0.02]
        if len(valid_positives) > 1:
            t_q_pos = traj.get(t_q, False)['obs']['eef_pos']
            n_vp = [v for v in valid_positives if np.linalg.norm(traj.get(v, False)['obs']['eef_pos'] - t_q_pos) > 0.1]
            valid_positives = n_vp if len(n_vp) else valid_positives
        valid_positives = valid_positives if len(valid_positives) == 1 else [v for v in valid_positives if v != t_q]
        t_p = int(np.random.choice(valid_positives))
        t_n = int(np.random.choice([t for t in range(len(traj)) if np.linalg.norm(obj_pos[t] - obj_pos[t_q]) > 0.075]))
        
        img_q, img_p, img_n = [resize(crop(traj[fr]['obs']['image'], self._crop), self._ctxt_dims, False) for fr in (t_q, t_p, t_n)]
        img_q, img_p, img_n  = [randomize_video([img], color_jitter=self._color_jitter, rand_gray=self._rand_gray, 
                                    rand_crop=self._rand_crop, normalize=self._normalize)[0] for img in (img_q, img_p, img_n)]
        img_q, img_p, img_n = [img.transpose((2, 0, 1)) for img in (img_q, img_p, img_n)]

        pos = np.concatenate((traj[t_q]['obs']['ee_aa'], traj[t_q]['obs']['gripper_qpos'][:1]))if is_agent else np.zeros((8,))
        return img_q, img_p, img_n, {'is_agent': is_agent, 'pos': pos.astype(np.float32)}
