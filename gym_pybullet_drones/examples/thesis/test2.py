from ray.rllib.algorithms.ppo import PPOConfig
from gym_pybullet_drones.envs.multi_agent_rl.CustomBaseMAA3 import CustomRl3
import os
import random
from ray.rllib.policy.policy import PolicySpec

def gen_policy(i):
    if bool(os.environ.get("RLLIB_ENABLE_RL_MODULE", False)):
        # just change the gammas between the two policies.
        # changing the module is not a critical part of this example.
        # the important part is that the policies are different.
        config = {
            "gamma": random.choice([0.95, 0.99]),
        }
    else:
        config = PPOConfig.overrides(
            # model={
            #     "custom_model": "model1",
            # },
            gamma=random.choice([0.95, 0.99]),
        )
    return PolicySpec(config=config)


# Setup PPO with an ensemble of `num_policies` different policies.
policies = {"policy_{}".format(i): gen_policy(i) for i in range(1)}
policy_ids = list(policies.keys())


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    pol_id = random.choice(policy_ids)
    return pol_id

algo =  PPOConfig()\
        .environment(CustomRl3)\
        .framework("torch")\
        .training(num_sgd_iter=5)\
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)\
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))).build()

# Get weights of the default local policy

print("###############################################################################################")
print("###############################################################################################")
print("###############################################################################################")
print("##############HERE WEIGHTS#######" , algo.get_policy())
# # Same as above
# algo.workers.local_worker().policy_map["default_policy"].get_weights()
#
# # Get list of weights of each worker, including remote replicas
# algo.workers.foreach_worker(lambda worker: worker.get_policy().get_weights())
#
# # Same as above, but with index.
# algo.workers.foreach_worker_with_id(
#     lambda _id, worker: worker.get_policy().get_weights()