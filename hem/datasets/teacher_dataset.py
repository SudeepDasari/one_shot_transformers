from .agent_dataset import AgentDemonstrations
from hem.datasets.util import resize
import random


class TeacherDemonstrations(AgentDemonstrations):
    def _get_pair(self, traj):
            pair = {}
            i = random.randint(0, len(traj) - self._T_pair)
            for j in range(self._T_pair):
                t = traj.get(j + i)
                pair['s_{}'.format(j)] = dict(image=resize(t['obs']['image'], self._im_dims))
            t = traj.get(self._T_pair + i)
            pair['s_{}'.format(self._T_pair)] = dict(image=resize(t['obs']['image'], self._im_dims))
            return pair
