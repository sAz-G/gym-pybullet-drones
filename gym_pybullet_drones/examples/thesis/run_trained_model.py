import time
import argparse
import numpy as np
import torch

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.CustomBaseMAA3 import CustomRl3
from ray.rllib.policy.policy import Policy

DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 30
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

# initial_positions =  np.array([
#     [1.0, -2.4492935982947064e-16, 1.0],
#     [0.8090169943749476, 0.9877852522924729, 1.0],
#     [-1.0, 1.2246467991473532e-16, 1.0],
#     [-0.8090169943749475, -0.987785252292473, 1.0],
#     ])
#
# set_of_targets = np.array([
#     [-1.0, 1.2246467991473532e-16, 1.0],
#     [-0.8090169943749475, -0.987785252292473, 1.0],
#     [1.0, -2.4492935982947064e-16, 1.0],
#     [0.8090169943749476, 0.9877852522924729, 1.0],
#     ])


initial_positions =  np.array([
    [2.0, 0, 1.0],
    [0, 2.0, 1.0]
    ])

set_of_targets = np.array([
    [1.0,0, 1.0],
    [.0, 1.0,1.0]
    ])

def run(
        drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        plot=True,
        colab=DEFAULT_COLAB
    ):

    num_drones = 2
    env = CustomRl3(
                        num_drones=num_drones,
                        set_of_positions=initial_positions,
                        set_of_targets=set_of_targets,
                        gui=True
                     )


    action = {k : np.zeros(4) for k in range(num_drones)}

    #checkpoint_path = "C:\\Users\sAz\\ray_results\PPO_2023-10-06_11-24-17\PPO_CustomRl3_1f998_00000_0_2023-10-06_11-24-17\checkpoint_000000\policies\policy_0"
    #checkpoint_path = "C:\\Users\sAz\\ray_results\PPO_2023-10-06_12-06-34\PPO_CustomRl3_07815_00000_0_2023-10-06_12-06-34\checkpoint_000000\policies\policy_0"
    checkpoint_path = "C:\\Users\sAz\\ray_results\PPO_2023-10-06_16-31-11\PPO_CustomRl3_fef36_00000_0_2023-10-06_16-31-11\checkpoint_000000\policies\policy_0"

    policy = Policy.from_checkpoint(checkpoint_path)
    START = time.time()

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        for k in range(num_drones):
            obs_temp = obs[k]
            action_temp = policy.compute_single_action(obs_temp)
            action_temp =  action_temp[0]
            action[k] = action_temp
        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

if __name__ == '__main__':
    run()

    # checkpoint_path = "C:\\Users\sAz\Documents\GitHub\gym-pybullet-drones\gym_pybullet_drones\examples\\results\latest\PPO\PPO_CustomRl3_de172_00000_0_2023-09-24_21-45-08\checkpoint_000100\policies\policy_0"
    # policy = Policy.from_checkpoint(checkpoint=checkpoint_path)
    #
    # print(type(policy))
    # wghts = policy.get_weights()
    #
    # print(wghts['pi.net.mlp.0.weight'].shape)
    #
    # temp_env = CustomRl3()
    # print(temp_env._observationSpace())
    #
    # print(temp_env._actionSpace())
    # print(wghts['encoder.actor_encoder.net.mlp.0.weight'].shape)
    # print(wghts['encoder.actor_encoder.net.mlp.2.weight'].shape)
    # print(wghts['pi.net.mlp.0.weight'].shape)
    # print(wghts.keys())
