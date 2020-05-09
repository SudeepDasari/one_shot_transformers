from torch.utils.data import Dataset
from .agent_dataset import AgentDemonstrations, SHUFFLE_RNG
from .teacher_dataset import TeacherDemonstrations
from hem.datasets import get_files
import torch
import os
import numpy as np
import random


class _AgentDatasetNoContext(AgentDemonstrations):
    def __init__(self, **params):
        params.pop('T_context', None)
        super().__init__(T_context=0, **params)


class ImitationDataset(Dataset):
    def __init__(self, root_dir, mode='train', split=[0.9, 0.1], **params):
        assert all([0 <= s <=1 for s in split]) and sum(split)  == 1, "split not valid!"
        agent_files, teacher_files = get_files(os.path.join(root_dir, 'traj*_robot')), get_files(os.path.join(root_dir, 'traj*_human'))
        order = [i for i in range(len(agent_files))]
        pivot = int(len(order) * split[0])
        if mode == 'train':
            order = order[:pivot]
        else:
            order = order[pivot:]
        random.Random(SHUFFLE_RNG).shuffle(order)
        agent_files = [agent_files[o] for o in order]
        teacher_files = [teacher_files[o] for o in order]
        assert len(teacher_files) == len(agent_files), "length of teacher files must match agent files!"
        self._teacher_dataset = TeacherDemonstrations(files=teacher_files, **params)
        self._agent_dataset = _AgentDatasetNoContext(files=agent_files, **params)

    def __len__(self):
        return len(self._agent_dataset)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        agent_traj = self._agent_dataset.get_traj(index)
        obj_detected = np.concatenate([agent_traj[t]['obs']['object_detected'] for t in range(len(agent_traj))])
        qpos = np.concatenate([agent_traj[t]['obs']['gripper_qpos'] for t in range(len(agent_traj))])
        if obj_detected.any():
            grip = int(np.argmax(obj_detected))
            drop = min(len(agent_traj) - 1, int(len(agent_traj) - np.argmax(obj_detected[::-1])))
        else:
            closed = np.isclose(qpos, 0)
            grip = int(np.argmax(closed))
            drop = min(len(agent_traj) - 1, int(len(agent_traj) - np.argmax(closed[::-1])))
        grip = np.concatenate((agent_traj[grip]['obs']['ee_pos'][:3], agent_traj[grip]['obs']['axis_angle'])).astype(np.float32)
        drop = np.concatenate((agent_traj[drop]['obs']['ee_pos'][:3], agent_traj[drop]['obs']['axis_angle'])).astype(np.float32)

        agent_pairs, _ = self._agent_dataset.proc_traj(agent_traj)
        agent_pairs['grip_location'], agent_pairs['drop_location'] = grip, drop
        teacher_context = self._teacher_dataset[index]
        return teacher_context, agent_pairs
