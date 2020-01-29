"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import gym
import ray
from ray import tune
from ray.rllib.utils import try_import_torch
from ray.tune import grid_search
from ray.tune.registry import register_env
from hem.robosuite.baxter.baxter_pick_place import BaxterPickPlaceCan


torch = try_import_torch
class EnvWrapper(BaxterPickPlaceCan, gym.Env):
    def __init__(self, config):
        super().__init__(**config)


class CustomModel(FullyConnectedNetwork):
    def forward(self, input_dict, state, seq_lens):
        import pdb; pdb.set_trace()
        return super().forward(input_dict, state, seq_lens)


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))

    ray.init(local_mode=True)
    ModelCatalog.register_custom_model("my_model", CustomModel)
    tune.run(
        "A3C",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": EnvWrapper,  # or "corridor" if registered above
            "model": {
                "custom_model": "my_model",
            },
            "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "sample_async": False,
            "env_config": {
                "use_camera_obs": False,
                "reward_shaping": True
            },
            "use_pytorch": True
        },
    )