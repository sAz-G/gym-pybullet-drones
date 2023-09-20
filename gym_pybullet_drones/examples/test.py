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

tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()

parser.add_argument("--num-agents", type=int, default=3)
parser.add_argument("--num-policies", type=int, default=1)
parser.add_argument("--num-cpus", type=int, default=16)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=200, help="Number of iterations to train." # was 200
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1000000, help="Number of timesteps to train." # was 100000
)
parser.add_argument(
    "--stop-reward", type=float, default=150.0, help="Reward at which we stop training."
)

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    # Register the models to use.


    temp_env = CustomRl3()
    pol = PolicySpec(observation_space=temp_env._observationSpace(),action_space=temp_env._actionSpace())
    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {"policy_{}".format(i): pol for i in range(args.num_policies)}
    policy_ids = list(policies.keys())


    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = random.choice(policy_ids)
        return pol_id

    config = (
        PPOConfig()
        .environment(CustomRl3)
        .framework(args.framework)
        .training(num_sgd_iter=20)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))

    )



    stop = {
      #  "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    results = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1, storage_path="./")
    ).fit()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()