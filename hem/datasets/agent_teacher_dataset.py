from torch.utils.data import Dataset
from .agent_dataset import AgentDemonstrations
from .teacher_dataset import TeacherDemonstrations
import torch
import os
from hem.datasets.util import resize
import numpy as np
import cv2


class AgentTeacherDataset(Dataset):
    def __init__(self, agent_dir, teacher_dir, **params):
        self._agent_dataset = AgentDemonstrations(agent_dir, **params)
        self._teacher_dataset = TeacherDemonstrations(teacher_dir, **params)
    
    def __len__(self):
        return len(self._agent_dataset) * len(self._teacher_dataset)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        a_idx = index % len(self._agent_dataset)
        t_idx = int(index // len(self._agent_dataset))

        agent_pairs, agent_context = self._agent_dataset[a_idx]
        teacher_context = self._teacher_dataset[t_idx]
        return teacher_context, agent_context, agent_pairs


class PairedAgentTeacherDataset(Dataset):
    def __init__(self, root_dir, pretend_unpaired=False, color_jitter=None, rand_crop=None, rand_rotate=None, is_rad=False, rand_translate=None, rand_gray=None, normalize=True, **params):
        self._agent_dataset = AgentDemonstrations(os.path.join(root_dir, 'traj*_robot.pkl'), normalize=False, **params)
        self._teacher_dataset = TeacherDemonstrations(os.path.join(root_dir, 'traj*_human.pkl'), normalize=False, **params)
        assert pretend_unpaired or len(self._agent_dataset) == len(self._teacher_dataset), "Lengths must match if data is paired!"

        self._pretend_unpaired = pretend_unpaired
        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_rot = rand_rotate if rand_rotate is not None else 0
        if not is_rad:
            self._rand_rot = np.radians(self._rand_rot)
        self._rand_trans = np.array(rand_translate if rand_translate is not None else [0, 0])
        self._rand_gray = rand_gray
        self._normalize = normalize

    def __len__(self):
        if self._pretend_unpaired:
            return len(self._agent_dataset) + len(self._teacher_dataset)
        return len(self._agent_dataset)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        if self._pretend_unpaired:
            if index < len(self._agent_dataset):
                context = self._agent_dataset[index][1]
            else:
                context = self._teacher_dataset[index - len(self._agent_dataset)]
            ctx1 = self._randomize(context.copy())
            return ctx1, self._randomize(context)
        
        # pairs aren't normalized, fix later for imitation learning
        _, agent_context = self._agent_dataset[index]
        teacher_context = self._teacher_dataset[index]
        return self._randomize(teacher_context), self._randomize(agent_context)
    
    def _randomize(self, frames):
        np.random.seed()
        frames = [fr for fr in np.transpose(frames, (0, 2, 3, 1))]
        
        if self._color_jitter is not None:
            rand_h, rand_s, rand_v = [np.random.uniform(-h, h) for h in self._color_jitter]
            delta = np.array([rand_h * 180, rand_s, rand_v]).reshape((1, 1, 3)).astype(np.float32)
            frames = [np.clip(cv2.cvtColor(cv2.cvtColor(fr, cv2.COLOR_RGB2HSV) + delta, cv2.COLOR_HSV2RGB), 0, 255) for fr in frames]
        if self._rand_gray and np.random.uniform() < self._rand_gray:
            frames = [np.tile(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)[:,:,None], (1,1,3)) for fr in frames]
        if self._rand_crop is not None:
            r, c = [min(np.random.randint(p + 1), m-10) for p, m in zip(self._rand_crop, frames[0].shape[:2])]
            if r:
                pad_r = np.zeros((r, frames[0].shape[1], 3)).astype(frames[0].dtype)
                if np.random.uniform() < 0.5:
                    frames = [np.concatenate((pad_r, fr[r:]), 0) for fr in frames]
                else:
                    frames = [np.concatenate((fr[:-r], pad_r), 0) for fr in frames]
            if c:                      
                pad_c = np.zeros((frames[0].shape[0], c, 3)).astype(frames[0].dtype)
                if np.random.uniform() < 0.5:
                    frames = [np.concatenate((pad_c, fr[:,c:]), 1) for fr in frames]
                else:
                    frames = [np.concatenate((fr[:,:-c], pad_c), 1) for fr in frames]
        if self._rand_rot or any(self._rand_trans):
            rot = np.random.uniform(-self._rand_rot, self._rand_rot)
            trans = np.random.uniform(-self._rand_trans, self._rand_trans)
            M = np.array([[np.cos(rot), -np.sin(rot), trans[0]], [np.sin(rot), np.cos(rot), trans[1]]])
            frames = [cv2.warpAffine(fr, M, (fr.shape[1], fr.shape[0])) for fr in frames]
        if self._normalize:
            frames = [resize(fr, (fr.shape[1], fr.shape[0]), True) for fr in frames]
        else:
            for fr in frames:
                fr /= 255
        frames = np.concatenate([fr[None] for fr in frames], 0).astype(np.float32)
        return np.transpose(frames, (0, 3, 1, 2))


class LabeledAgentTeacherDataset(PairedAgentTeacherDataset):
    def __init__(self, root_dir, color_jitter=None, rand_crop=None, rand_rotate=None, is_rad=False, rand_translate=None, rand_gray=None, normalize=True, **params):
        self._agent_dataset = AgentDemonstrations(os.path.join(root_dir, 'traj*_robot.pkl'), normalize=False, **params)
        self._teacher_dataset = TeacherDemonstrations(os.path.join(root_dir, 'traj*_human.pkl'), normalize=False, **params)

        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_rot = rand_rotate if rand_rotate is not None else 0
        if not is_rad:
            self._rand_rot = np.radians(self._rand_rot)
        self._rand_trans = np.array(rand_translate if rand_translate is not None else [0, 0])
        self._rand_gray = rand_gray
        self._normalize = normalize
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        if index < len(self._agent_dataset):
            context = self._agent_dataset[index][1]
        else:
            context = self._teacher_dataset[index - len(self._agent_dataset)]
        ctx1 = self._randomize(context.copy())
        return ctx1, int(index < len(self._agent_dataset)), self._randomize(context)
