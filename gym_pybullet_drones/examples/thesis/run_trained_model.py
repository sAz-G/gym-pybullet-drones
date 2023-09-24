import time
import argparse
import numpy as np
import torch

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.CustomBaseMAA3 import CustomRl3
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.ppo import PPOTorchPolicy

DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

initial_positions =  np.array([
    [1.0, -2.4492935982947064e-16, 1.0],
    [0.8090169943749476, 0.9877852522924729, 1.0],
    [-1.0, 1.2246467991473532e-16, 1.0],
    [-0.8090169943749475, -0.987785252292473, 1.0],
    ])

set_of_targets = np.array([
    [-1.0, 1.2246467991473532e-16, 1.0],
    [-0.8090169943749475, -0.987785252292473, 1.0],
    [1.0, -2.4492935982947064e-16, 1.0],
    [0.8090169943749476, 0.9877852522924729, 1.0],
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
    env = CustomRl3(
                        num_drones=4,
                        set_of_positions=initial_positions,
                        set_of_targets=set_of_targets,
                     )


    action = { k : np.zeros(4) for k in range(4)}

    checkpoint_path = "C:\\Users\sAz\Documents\GitHub\gym-pybullet-drones\gym_pybullet_drones\examples\\results\latest\PPO\PPO_CustomRl3_e2b4b_00000_0_2023-09-24_15-04-24\checkpoint_000002\policies\policy_0"


    policy = Policy.from_checkpoint(checkpoint_path)
    START = time.time()

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        for k in range(4):
            obs_temp = obs[k]
            action_temp = policy.compute_single_action(obs_temp)
            print( action_temp)
            action_temp =  np.abs(action_temp[0])
            action[k] = action_temp
        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

if __name__ == '__main__':
    #run()

    checkpoint_path = "C:\\Users\sAz\Documents\GitHub\gym-pybullet-drones\gym_pybullet_drones\examples\\results\latest\PPO\PPO_CustomRl3_20115_00000_0_2023-09-24_15-13-17\checkpoint_000002\policies\policy_0"


    policy = Policy.from_checkpoint(checkpoint=checkpoint_path)

    print(type(policy))
    wghts = policy.get_weights()

    print(wghts['pi.net.mlp.0.weight'].shape)

    temp_env = CustomRl3()
    print(temp_env._observationSpace())

    print(temp_env._actionSpace())
    print(wghts['encoder.actor_encoder.net.mlp.0.weight'].shape)
    print(wghts['encoder.actor_encoder.net.mlp.2.weight'].shape)
    print(wghts['pi.net.mlp.0.weight'].shape)
    #print(wghts.keys())

    policy.export_model("./")
    chkpnt = torch.load("C:\\Users\sAz\Documents\GitHub\gym-pybullet-drones\gym_pybullet_drones\examples\\thesis\model.pt")
    print(chkpnt)