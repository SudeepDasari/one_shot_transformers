from torch.utils.data import Dataset
from .agent_dataset import AgentDemonstrations
from .teacher_dataset import TeacherDemonstrations
from hem.datasets import load_traj
from hem.datasets.util import randomize_video, split_files
import torch
import os
import numpy as np
import json
import pickle as pkl


class _AgentDatasetNoContext(AgentDemonstrations):
    def __init__(self, **params):
        params.pop('T_context', None)
        super().__init__(T_context=0, **params)


class ImitationDataset(Dataset):
    def __init__(self, root_dir, mode='train', split=[0.9, 0.1], before_grip=False, recenter_actions=False, **params):
        self._root = os.path.expanduser(root_dir)
        mappings_file = os.path.join(self._root, 'mappings.json')
        with open(mappings_file, 'r') as f:
            self._mappings = json.load(f)
        
        teacher_files = sorted(list(self._mappings.keys()))
        order = split_files(len(teacher_files), split, mode)
        self._teacher_files = [teacher_files[o] for o in order]
        self._teacher_dataset = TeacherDemonstrations(files=[], **params)
        self._agent_dataset = _AgentDatasetNoContext(files=[], **params)
        self._before_grip = before_grip
        self._recenter_actions = recenter_actions

    def __len__(self):
        return len(self._teacher_files)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        # retrieve trajectory from mapping
        teacher_traj, agent_traj = self._teacher_files[index], self._mappings[self._teacher_files[index]]
        teacher_traj, agent_traj = [load_traj(os.path.join(self._root, f_name)) for f_name in (teacher_traj, agent_traj)]

        obj_detected = np.concatenate([agent_traj.get(t, False)['obs']['object_detected'] for t in range(len(agent_traj))])
        qpos = np.concatenate([agent_traj.get(t, False)['obs']['gripper_qpos'] for t in range(len(agent_traj))])
        if obj_detected.any():
            grip_t = int(np.argmax(obj_detected))
            drop_t = min(len(agent_traj) - 1, int(len(agent_traj) - np.argmax(obj_detected[::-1])))
        else:
            closed = np.isclose(qpos, 0)
            grip_t = int(np.argmax(closed))
            drop_t = min(len(agent_traj) - 1, int(len(agent_traj) - np.argmax(closed[::-1])))
        grip, drop = agent_traj.get(grip_t, False), agent_traj.get(drop_t, False)
        grip = np.concatenate((grip['obs']['ee_pos'][:3], grip['obs']['axis_angle'])).astype(np.float32)
        drop = np.concatenate((drop['obs']['ee_pos'][:3], drop['obs']['axis_angle'])).astype(np.float32)

        if self._before_grip: # make this hack more elegant
            agent_pairs = self._agent_dataset._get_pairs(agent_traj, grip_t)
        else:
            agent_pairs, _ = self._agent_dataset.proc_traj(agent_traj)
        agent_pairs['grip_location'], agent_pairs['drop_location'] = grip, drop

        if self._recenter_actions:
            mean = np.array([0.647, 0.0308, 0.10047, 1, 0.1464, 0.1464, 0.010817]).reshape((1, -1))
            std = np.array([0.231, 0.447, 0.28409, 0.04, 0.854, 0.854, 0.0653]).reshape((1, -1))
            agent_pairs['actions'][:,:7] -= mean.astype(np.float32)
            agent_pairs['actions'][:,:7] /= std.astype(np.float32)
            for k in ('grip_location', 'drop_location'):
                agent_pairs[k] -= mean[0]
                agent_pairs[k] /= std[0]
        return self._teacher_dataset.proc_traj(teacher_traj), agent_pairs


class StateDataset(Dataset):
    def __init__(self, state_file, min_T=50, max_T=300, center=False, mode='train', split=[0.9, 0.1]):
        self._all_trajs = pkl.load(open(os.path.expanduser(state_file), 'rb'))
        order = split_files(len(self._all_trajs), split, mode)
        self._all_trajs = [self._all_trajs[o] for o in order]
        self._min_T = min_T
        self._max_T = max_T
        self._center = center
    
    def __len__(self):
        return len(self._all_trajs)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        all_states, all_actions = self._all_trajs[index]['states'][:-1], self._all_trajs[index]['actions']
        seq_len = np.random.randint(self._min_T, self._max_T)
        start = np.random.randint(len(all_actions) - seq_len + 1) if len(all_actions) >= seq_len else 0

        states = all_states[start:start+seq_len]
        actions = all_actions[start:start+seq_len]
        assert len(states) == len(actions), "action/state lengths don't match, bad striding?"
        x_len = len(actions)
        loss_mask = np.array([1 if i < x_len else 0 for i in range(self._max_T)])
        
        # pad states and actions
        states = np.concatenate((states, np.zeros((self._max_T - x_len, states.shape[1])))).astype(np.float32)
        actions = np.concatenate((actions, np.zeros((self._max_T - x_len, actions.shape[1])))).astype(np.float32)
        
        if self._center:
            mean = np.array([0.647, 0.0308, 0.10047, 1, 0.1464, 0.1464, 0.010817]).reshape((1, -1))
            std = np.array([0.231, 0.447, 0.28409, 0.04, 0.854, 0.854, 0.0653]).reshape((1, -1))
            for tensor in [states, actions]:
                tensor[:,:7] -= mean.astype(np.float32)
                tensor[:,:7] /= std.astype(np.float32)
        
        return states, actions, x_len, loss_mask
