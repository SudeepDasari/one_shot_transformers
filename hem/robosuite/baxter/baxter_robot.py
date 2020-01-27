from robosuite.models.robots.baxter_robot import Baxter
import numpy as np


class BaxterRobot(Baxter):
    @property
    def init_qpos(self):
        return np.array([
            0.535, -0.093, 0.038, 0.166, 0.643, 1.960, -1.297,
            1.703, -0.026, -0.076, 0.175, -0.748, 1.641, -0.158])
