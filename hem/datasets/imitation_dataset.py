from torch.utils.data import Dataset
from .agent_dataset import AgentDemonstrations
from .teacher_dataset import TeacherDemonstrations
from hem.datasets import load_traj
from hem.datasets.util import randomize_video, split_files, resize
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
    def __init__(self, state_file, min_T=50, max_T=300, center=False, mode='train', split=[0.9, 0.1], ret_start_T=False):
        self._all_trajs = pkl.load(open(os.path.expanduser(state_file), 'rb'))
        self._order = split_files(len(self._all_trajs), split, mode)
        self._all_trajs = [self._all_trajs[o] for o in self._order]
        self._min_T = min_T
        self._max_T = max_T
        self._center = center
        self._ret_start_T = ret_start_T
    
    def __len__(self):
        return len(self._all_trajs)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        all_states, all_actions = self._all_trajs[index]['states'][:-1], self._all_trajs[index]['actions']
        seq_len = np.random.randint(self._min_T, self._max_T)
        start = np.random.randint(len(all_actions) - seq_len + 1) if len(all_actions) >= seq_len else 0

        grip_pose = self._all_trajs[index]['states'][self._all_trajs[index]['grip_t'],:7].copy()[None]
        drop_pose = self._all_trajs[index]['states'][self._all_trajs[index]['drop_t'],:7].copy()[None]
        aux_mask = np.ones(14).astype(np.float32)
        if not (start <= self._all_trajs[index]['grip_t'] < start + seq_len):
            aux_mask[:7] *= 0
        if not (start <= self._all_trajs[index]['drop_t'] < start + seq_len):
            aux_mask[7:] *= 0

        states = all_states[start:start+seq_len].copy()
        actions = all_actions[start:start+seq_len].copy()
        assert len(states) == len(actions), "action/state lengths don't match, bad striding?"
        x_len = len(actions)
        loss_mask = np.array([1 if i < x_len else 0 for i in range(self._max_T)])
        
        # pad states and actions
        states = np.concatenate((states, np.zeros((self._max_T - x_len, states.shape[1])))).astype(np.float32)
        actions = np.concatenate((actions, np.zeros((self._max_T - x_len, actions.shape[1])))).astype(np.float32)
        
        if self._center:
            mean = np.array([0.647, 0.0308, 0.10047, 1, 0.1464, 0.1464, 0.010817]).reshape((1, -1))
            std = np.array([0.231, 0.447, 0.28409, 0.04, 0.854, 0.854, 0.0653]).reshape((1, -1))
            for tensor in [states, actions, grip_pose, drop_pose]:
                tensor[:x_len,:7] -= mean.astype(np.float32)
                tensor[:x_len,:7] /= std.astype(np.float32)
        
        if self._ret_start_T:
            return states, actions, x_len, loss_mask.astype(np.float32), np.concatenate((grip_pose[0], drop_pose[0])).astype(np.float32), aux_mask, start
        return states, actions, x_len, loss_mask.astype(np.float32), np.concatenate((grip_pose[0], drop_pose[0])).astype(np.float32), aux_mask


class StateDatasetVisionContext(StateDataset):
    def __init__(self, state_file, img_dir, img_width=320, img_height=240, rand_crop=None, color_jitter=None, rand_gray=None, **kwargs):
        super().__init__(state_file, ret_start_T=True, **kwargs)
        self._img_dir = os.path.expanduser(img_dir)
        self._img_height, self._img_width = img_height, img_width
        self._human_grip_times = json.load(open(os.path.join(self._img_dir, 'human_grip_timings.json'), 'r'))
        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_gray = rand_gray
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        s, a, lens, lm, aux, auxm, start_T = super().__getitem__(index)

        traj_ind = self._order[index]
        robot_frame = load_traj(os.path.join(self._img_dir, 'traj{}_robot.pkl'.format(traj_ind)))[start_T]['obs']['image']
        human_vid = load_traj(os.path.join(self._img_dir, 'traj{}_human.pkl'.format(traj_ind)))
        grip_time = self._human_grip_times['traj{}_human.pkl'.format(traj_ind)]
        start_time = 0 if grip_time < 8 else np.random.randint(6)
        end_time = len(human_vid) - np.random.randint(1, 6)
        start, mid, end = [human_vid[t]['obs']['image'] for t in (start_time, grip_time, end_time)]

        context = [resize(i, (self._img_width, self._img_height), False) for i in (robot_frame, start, mid, end)]
        context = randomize_video(context, color_jitter=self._color_jitter, rand_gray=self._rand_gray, rand_crop=self._rand_crop, normalize=True).transpose((0, 3, 1, 2))
        return s, a, lens, lm, aux, auxm, context


class AuxDataset(Dataset):
    def __init__(self, root_dir, img_width=320, img_height=240, rand_crop=None, color_jitter=None, rand_gray=None, split=[0.9, 0.1], mode='train'):
        self._img_dir = os.path.expanduser(root_dir)
        self._img_height, self._img_width = img_height, img_width
        self._human_grip_times = json.load(open(os.path.join(self._img_dir, 'human_grip_timings.json'), 'r'))
        self._file_inds = split_files(len(self._human_grip_times), split, mode)
        self._color_jitter = color_jitter
        self._rand_crop = rand_crop
        self._rand_gray = rand_gray
    
    def __len__(self):
        return len(self._file_inds)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        traj_ind = self._file_inds[index]
        robot_traj = load_traj(os.path.join(self._img_dir, 'traj{}_robot.pkl'.format(traj_ind)))
        robot_frame = robot_traj[0]['obs']['image']
        obj_detected = np.concatenate([robot_traj.get(t, False)['obs']['object_detected'] for t in range(len(robot_traj))])
        qpos = np.concatenate([robot_traj.get(t, False)['obs']['gripper_qpos'] for t in range(len(robot_traj))])
        if obj_detected.any():
            grip_t = int(np.argmax(obj_detected))
            drop_t = min(len(robot_traj) - 1, int(len(robot_traj) - np.argmax(obj_detected[::-1])))
        else:
            closed = np.isclose(qpos, 0)
            grip_t = int(np.argmax(closed))
            drop_t = min(len(robot_traj) - 1, int(len(robot_traj) - np.argmax(closed[::-1])))
        grip, drop = robot_traj.get(grip_t, False), robot_traj.get(drop_t, False)
        grip = np.concatenate((grip['obs']['ee_pos'][:3], grip['obs']['axis_angle'])).astype(np.float32)
        drop = np.concatenate((drop['obs']['ee_pos'][:3], drop['obs']['axis_angle'])).astype(np.float32)
        for x in (grip, drop):
            if x[3] < 0:
                x[3] += 2
        
        human_vid = load_traj(os.path.join(self._img_dir, 'traj{}_human.pkl'.format(traj_ind)))
        grip_time = self._human_grip_times['traj{}_human.pkl'.format(traj_ind)]
        start_time = 0 if grip_time < 8 else np.random.randint(6)
        end_time = len(human_vid) - np.random.randint(1, 6)
        start, mid, end = [human_vid[t]['obs']['image'] for t in (start_time, grip_time, end_time)]

        context = [resize(i, (self._img_width, self._img_height), False) for i in (robot_frame, start, mid, end)]
        context = randomize_video(context, color_jitter=self._color_jitter, rand_gray=self._rand_gray, rand_crop=self._rand_crop, normalize=True).transpose((0, 3, 1, 2))
        return context, np.concatenate((grip, drop)).astype(np.float32)
