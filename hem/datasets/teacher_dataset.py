from .agent_dataset import AgentDemonstrations
from hem.datasets.util import resize
import random


class TeacherDemonstrations(AgentDemonstrations):
    def __init__(self, teacher_dir,  **kwargs):
        super().__init__(teacher_dir, **kwargs)

    def _proc_traj(self, traj):
        return self._make_context(traj)
