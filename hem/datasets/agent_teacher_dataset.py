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
    def __init__(self, root_dir, pretend_unpaired=False, **params):
        self._agent_dataset = AgentDemonstrations(os.path.join(root_dir, 'traj*_robot.pkl'), **params)
        self._teacher_dataset = TeacherDemonstrations(os.path.join(root_dir, 'traj*_human.pkl'), **params)
        assert pretend_unpaired or len(self._agent_dataset) == len(self._teacher_dataset), "Lengths must match if data is paired!"
        self._pretend_unpaired = pretend_unpaired

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
                ctx1, ctx2 = [self._agent_dataset[index][1] for _ in range(2)]
            else:
                ctx1, ctx2 = [self._teacher_dataset[index - len(self._agent_dataset)] for _ in range(2)]
            return ctx1, ctx2
        
        # pairs aren't returned, fix later for imitation learning?
        _, agent_context = self._agent_dataset[index]
        teacher_context = self._teacher_dataset[index]
        return teacher_context, agent_context


class LabeledAgentTeacherDataset(PairedAgentTeacherDataset):
    def __init__(self, root_dir, ignore_actor=False, **params):
        self._agent_dataset = AgentDemonstrations(os.path.join(root_dir, 'traj*_robot.pkl'), **params)
        self._teacher_dataset = TeacherDemonstrations(os.path.join(root_dir, 'traj*_human.pkl'), **params)
        self._ignore_actor = ignore_actor

    def __len__(self):
        if self._ignore_actor == 'agent':
            return len(self._teacher_dataset)
        if self._ignore_actor == 'teacher':
            return len(self._agent_dataset)
        return len(self._agent_dataset) + len(self._teacher_dataset)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        if self._ignore_actor == 'agent':
            ctx1, ctx2 = [self._teacher_dataset[index] for _ in range(2)]
        elif self._ignore_actor == 'teacher':
            ctx1, ctx2  = [self._agent_dataset[index][1] for _ in range(2)]
        else:
            if index < len(self._agent_dataset):
                ctx1, ctx2  = [self._agent_dataset[index][1] for _ in range(2)]
            else:
                ctx1, ctx2  = [self._teacher_dataset[index - len(self._agent_dataset)] for _ in range(2)]
        
        return ctx1, int(index < len(self._agent_dataset)), ctx2
