"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from gym import Env, spaces
import ray
from ray import tune
from ray.rllib.utils import try_import_torch
from ray.tune import grid_search
from ray.tune.registry import register_env
from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceCan
from collections import OrderedDict


class EnvWrapper(BaxterPickPlaceCan, Env):
    def __init__(self, config):
        super().__init__(**config)
        self.action_space = spaces.Box(-1., 1., shape=(8,), dtype='float32')
        self.observation_space = OrderedDict()
        for k, v in self._get_observation().items():
            self.observation_space[k] = spaces.Box(-np.inf, np.inf, shape=v.shape, dtype='float32')
        self.observation_space = spaces.Dict(self.observation_space)


class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    # \lambda \in [0.9, 1]

    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": EnvWrapper,  # or "corridor" if registered above
            "num_workers": 4,
            "model": {
                "custom_model": "my_model",
            },
            "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "sample_async": False,
            "env_config": {
                "use_camera_obs": False,
                "reward_shaping": True
            },
        },
    )