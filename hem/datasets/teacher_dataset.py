from .agent_dataset import AgentDemonstrations
from hem.datasets.util import resize
import random


class TeacherDemonstrations(AgentDemonstrations):
    def __init__(self, N_pair=0, **kwargs):
        assert N_pair == 0, "Cannot return (s,a) pairs!"
        super().__init__(N_pair=N_pair, **kwargs)

    def _proc_traj(self, traj):
        return self._make_context(traj)
