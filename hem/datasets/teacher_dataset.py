from .agent_dataset import AgentDemonstrations
from hem.datasets.util import resize
import random


class TeacherDemonstrations(AgentDemonstrations):
    def _proc_traj(self, traj):
        return self._make_context(traj)
