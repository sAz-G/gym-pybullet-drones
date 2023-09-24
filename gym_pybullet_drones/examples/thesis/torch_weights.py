import numpy as np
from ray.rllib.policy.policy import Policy
from ray import air, tune

checkpoint_path = "/gym_pybullet_drones/examples/results/latest/PPO/PPO_CustomRl3_e6ded_00000_0_2023-09-21_16-12-30/checkpoint_000001/policies/policy_0"




policy = Policy.from_checkpoint(checkpoint_path)

print(policy)
print(policy.action_space)
print(policy.observation_space)
print(policy.get_weights())

print(policy.compute_single_action(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])))
