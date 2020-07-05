from torch.utils.data import Dataset
from .agent_dataset import AgentDemonstrations
from .teacher_dataset import TeacherDemonstrations
from hem.datasets import get_files, load_traj
import torch
import os
import numpy as np
import random
import json
from hem.datasets.util import split_files


class AgentTeacherDataset(Dataset):
    def __init__(self, agent_dir, teacher_dir, agent_context=None, epoch_repeat=5, **params):
        teacher_context = params.pop('T_context', 15)
        self._agent_context = agent_context = agent_context if agent_context is not None else teacher_context
        self._agent_dataset = AgentDemonstrations(agent_dir, T_context=agent_context, **params)
        self._teacher_dataset = TeacherDemonstrations(teacher_dir, T_context=teacher_context, **params)
        assert len(self._agent_dataset) == len(self._teacher_dataset)
        self._epoch_repeat = epoch_repeat
    
    def __len__(self):
        return len(self._agent_dataset) * self._epoch_repeat
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        index = index % len(self._agent_dataset)
        np.random.seed()
        agent_pairs, agent_context = self._agent_dataset[index]
        teacher_context = self._teacher_dataset[index]
        
        if self._agent_context:
            return teacher_context, agent_context, agent_pairs
        return teacher_context, agent_pairs


class PairedAgentTeacherDataset(Dataset):
    def __init__(self, root_dir, mode='train', split=[0.9, 0.1], **params):
        self._root_dir = os.path.expanduser(root_dir)
        with open(os.path.join(self._root_dir, 'mappings_1_to_1.json'), 'r') as f:
            self._mapping = json.load(f)
        self._teacher_files = sorted(list(self._mapping.keys()))
        self._teacher_files = [self._teacher_files[o] for o in split_files(len(self._teacher_files), split, mode)]

        self._agent_dataset = AgentDemonstrations(files=[], **params)
        self._teacher_dataset = TeacherDemonstrations(files=[], **params)

    def __len__(self):
        return len(self._teacher_files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        assert 0 <= index < len(self), "invalid index!"

        teacher_traj = load_traj(os.path.join(self._root_dir, self._teacher_files[index]))
        agent_traj = load_traj(os.path.join(self._root_dir, self._mapping[self._teacher_files[index]]))
        _, agent_context = self._agent_dataset.proc_traj(agent_traj)
        teacher_context = self._teacher_dataset.proc_traj(teacher_traj)
        return teacher_context, agent_context


class LabeledAgentTeacherDataset(PairedAgentTeacherDataset):
    def __init__(self, root_dir, ignore_actor=False, **params):
        self._agent_dataset = AgentDemonstrations(os.path.join(root_dir, 'traj*_robot'), **params)
        self._teacher_dataset = TeacherDemonstrations(os.path.join(root_dir, 'traj*_human'), **params)
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
