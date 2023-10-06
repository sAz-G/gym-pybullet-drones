import numpy as np
from ray.rllib.policy.policy import Policy
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO


checkpoint_path = "C:\\Users\sAz\\ray_results\sharif_bivarl\PPO_2023-10-05_11-54-50\PPO_CustomRl3_3978c_00000_0_2023-10-05_11-54-50\checkpoint_000000\policies\policy_0"


#weights = algo.get_policy(policy_name).get_state()["weights"]

policy = Policy.from_checkpoint(checkpoint_path)

# print(policy)
# print(policy.action_space)
# print(policy.observation_space)
# print(policy.get_weights())
# print(policy.get_state()["weights"]['encoder.actor_encoder.net.mlp.0.weight'].shape)
# print(policy.get_state()["weights"]['encoder.actor_encoder.net.mlp.2.weight'].shape)
print(policy.get_state()["weights"]["pi.net.mlp.0.weight"].shape)
print(policy.get_state()["weights"]["pi.net.mlp.1.weight"].shape)

# print(policy.compute_single_action(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])))
