from torch.utils.data import Dataset
from .agent_dataset import AgentDemonstrations
from .teacher_dataset import TeacherDemonstrations
import torch


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
        _, teacher_context = self._teacher_dataset[t_idx]
        return teacher_context, agent_context, agent_pairs
