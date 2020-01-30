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
from collections import OrderedDict
import hem.robosuite
import tensorflow as tf


class FCNet(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,name):
        super(FCNet, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        
        self._hiddens = [128, 128, 64, 32]
        inputs = tf.keras.layers.Input(shape=(np.product(obs_space.shape),), name="observations")

        fc_top = self._build_fc_tower(inputs, 'fc')
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="fc_out",
            activation=None)(fc_top)
        
        value_top = self._build_fc_tower(inputs, 'value')
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None)(value_top)

        self._action_value_model = tf.keras.Model(inputs, [layer_out, value_out])
        self._value_out = None
        self.register_variables(self._action_value_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self._action_value_model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def _build_fc_tower(self, inputs, name):
        i, last_layer = 1, inputs
        for size in self._hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="{}_{}".format(name, i),
                activation=tf.nn.relu)(last_layer)
            i += 1
        return last_layer


if __name__ == "__main__":
    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    # \lambda \in [0.9, 1]

    ray.init()
    ModelCatalog.register_custom_model("my_model", FCNet)
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 2000000,
            "episode_reward_max": 200
        },
        config={
            "env": "RoboPickPlace",  # or "corridor" if registered above
            "model": {
                "custom_model": "my_model",
            },
            "env_config": {
                "use_camera_obs": False,
                "reward_shaping": True,
                "env_name": "SawyerPickPlaceCan",
                "obs_filters": ['robot-state',  'object-state']
            },
            "num_workers": 12,
            "sample_async": False,
            "lr": 2.5e-4,
            "vf_loss_coeff": 0.5,
            "lr_schedule": [(0, 2.5e-4), (100000, 5e-5), (3000000, 1e-5), (6000000, 1e-6)],
            "lambda": grid_search([0.9, 0.95, 0.99])
        },
        checkpoint_at_end=True,
        checkpoint_freq=100000,
    )
