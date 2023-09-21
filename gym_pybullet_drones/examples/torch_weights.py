from ray.rllib.policy.policy import Policy

checkpoint_path = "C:\\Users\sAz\Documents\GitHub\gym-pybullet-drones\gym_pybullet_drones\examples\PPO\PPO_CustomRl3_2bb19_00000_0_2023-09-20_11-07-49\checkpoint_000200\policies\policy_0"
policy = Policy.from_checkpoint(checkpoint_path)

print(policy)
print(policy.get_weights())