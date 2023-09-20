from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.tune.logger import pretty_print
from gym_pybullet_drones.envs.multi_agent_rl.CustomBaseMAA3 import CustomRl3
import os
import numpy as np
import ray
import random
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import MODEL_DEFAULTS

ray.init(num_cpus=5)


# Register the models to use.

# Each policy can have a different configuration (including custom model).
temp_env = CustomRl3()
pol = PolicySpec()
# Setup PPO with an ensemble of `num_policies` different policies.
policies = {"policy_{}".format(i): pol for i in range(1)}
policy_ids = list(policies.keys())


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    pol_id = random.choice(policy_ids)
    return pol_id

MODEL_DEFAULTS["fcnet_hiddens"] = [1,1]


algo = (
    PPOConfig()
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env=CustomRl3)
    .framework("torch")
    .training(num_sgd_iter=5)
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .update_from_dict({"model":{"fcnet_hiddens": [1],}})
    .build()
)

print(algo.get_policy())
print(algo)
print()

ray.shutdown()
if __name__ == '__main__':
    TRAIN = False

    if TRAIN:
        for i in range(10):
            result = algo.train()
            print(pretty_print(result))

            if i % 5 == 0:
                checkpoint_dir = algo.save()
                print(f"Checkpoint saved in directory {checkpoint_dir}")