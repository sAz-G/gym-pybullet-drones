"""Simple example of setting up a multi-agent policy mapping.

Control the number of agents and policies via --num-agents and --num-policies.

This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.

Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""

import argparse
import os
import random

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved

from gym_pybullet_drones.envs.multi_agent_rl.CustomBaseMAA3 import CustomRl3
from ray.rllib.policy.policy import Policy
import gymnasium as gym
from ray.tune.logger import pretty_print

if __name__ == "__main__":

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=0)
        .environment(env="CartPole-v1")
    ).build()

    temp_env = gym.make("CartPole-v1")

    print(temp_env.observation_space)
    print(temp_env.action_space)
    print(algo.get_policy().get_weights()['encoder.actor_encoder.net.mlp.0.weight'].shape)
    print(algo.get_policy().get_weights()['encoder.actor_encoder.net.mlp.2.weight'].shape)
    print(algo.get_policy().get_weights()['pi.net.mlp.0.weight'].shape)
    print(algo.get_policy().get_weights().keys())

