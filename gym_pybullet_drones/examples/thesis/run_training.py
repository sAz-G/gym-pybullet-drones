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


if __name__ == "__main__":

    stop_iter       = 2
    stop_timesteps  = 10**7
    stop_reward     = 150

    ray.init(num_cpus=16)

    temp_env = CustomRl3()
    pol = PolicySpec(action_space=temp_env._actionSpace(), observation_space=temp_env._observationSpace())
    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {"policy_{}".format(i): pol for i in range(1)}
    policy_ids = list(policies.keys())

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = random.choice(policy_ids)
        return pol_id

    config = (
        PPOConfig()
        .environment(CustomRl3)
        .framework("torch")
        #.training(num_sgd_iter=20)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .update_from_dict({"model":{"fcnet_hiddens": [256,256],}})
        #.resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .resources(num_gpus=1)
        .rollouts(num_rollout_workers=1)
    )

    stop = {
      #  "episode_reward_mean": stop_reward,
        "timesteps_total": stop_timesteps,
        "training_iteration": stop_iter,
    }

    pth = "C:\\Users\sAz\Documents\GitHub\gym-pybullet-drones\gym_pybullet_drones\examples\\results\latest\\"

    results = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1, storage_path=pth)
    ).fit()

    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)

    ray.shutdown()


