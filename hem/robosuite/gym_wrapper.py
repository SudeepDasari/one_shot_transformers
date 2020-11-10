from hem.robosuite import get_env
from gym import spaces, Env
from collections import OrderedDict
import numpy as np


class GymWrapper(Env):
    def __init__(self, config):
        env_class = get_env(config.pop('env_name'))
        self._obs_filters = config.pop('obs_filters', False)
        self._wrapped_env = env_class(**config)

        obs = self._wrapped_env._get_observation()
        self._observation_space = {}
        if isinstance(obs, OrderedDict):
            self._observation_space = OrderedDict()
        
        for k, v in obs.items():
            if self._obs_filters and k not in self._obs_filters:
                continue
            self._observation_space[k] = spaces.Box(-np.inf, np.inf, shape=v.shape, dtype='float32')
        
        if self._obs_filters:
            for k in self._obs_filters:
                assert k in self._observation_space, "key {} missing in environment".format(k)
        self._observation_space = spaces.Dict(self._observation_space)
    
    @property
    def action_space(self):
        return spaces.Box(-1., 1., shape=(self._wrapped_env.dof,), dtype='float32')
    
    @property
    def observation_space(self):
        return self._observation_space
    
    def step(self, action):
        o, r, d, i = self._wrapped_env.step(action)
        return self._pre_process_obs(o), r, d, i
    
    def reset(self):
        return self._pre_process_obs(self._wrapped_env.reset())
    
    def render(self, mode='N/A'):
        return self._wrapped_env.render()
    
    def close(self):
        return self._wrapped_env.close()

    def _pre_process_obs(self, obs):
        if self._obs_filters:
            for k in list(obs.keys()):
                if k not in self._obs_filters:
                    obs.pop(k)
        return obs


try:    
    from ray.tune.registry import register_env
    register_env("RoboPickPlace", lambda config: GymWrapper(config))
except:
    pass

