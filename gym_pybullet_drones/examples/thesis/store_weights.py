import numpy
from ray.rllib.policy.policy import Policy
from numpy import genfromtxt

def store_weights(checkpoint_path, store_path):

    policy = Policy.from_checkpoint(checkpoint_path)

    input_layer_weights   = policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.0.weight"]
    input_layer_bias      = policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.0.bias"]
    hidden0_layer_weights = policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.2.weight"]
    hidden0_layer_bias    = policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.2.bias"]
    hidden1_layer_weights = policy.get_state()["weights"]["pi.net.mlp.0.weight"]
    hidden1_layer_bias    = policy.get_state()["weights"]["pi.net.mlp.0.bias"]

    numpy.savetxt(store_path + '/input_layer_w.csv',   input_layer_weights,     delimiter=",")
    numpy.savetxt(store_path + '/input_layer_b.csv',   input_layer_bias,        delimiter=",")
    numpy.savetxt(store_path + '/hidden0_layer_w.csv', hidden0_layer_weights,   delimiter=",")
    numpy.savetxt(store_path + '/hidden0_layer_b.csv', hidden0_layer_bias,      delimiter=",")
    numpy.savetxt(store_path + '/hidden1_layer_w.csv', hidden1_layer_weights,   delimiter=",")
    numpy.savetxt(store_path + '/hidden1_layer_b.csv', hidden1_layer_bias,      delimiter=",")



if __name__ == '__main__':
    checkpoint_path = "C:\\Users\sAz\\ray_results\PPO_2023-10-06_11-24-17\PPO_CustomRl3_1f998_00000_0_2023-10-06_11-24-17\checkpoint_000000\policies\policy_0"
    store_path = "C:\\Users\sAz\Documents\GitHub\gym-pybullet-drones\gym_pybullet_drones\examples\\thesis\\nn"
    # policy = Policy.from_checkpoint(checkpoint_path)
    #
    # input_layer_weights = policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.0.weight"]
    # input_layer_bias = policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.0.weight"]
    # hidden0_layer_weights = policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.0.weight"]
    # hidden0_layer_bias = policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.0.weight"]
    # hidden1_layer_weights = policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.0.weight"]
    # hidden1_layer_bias = policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.0.weight"]
    #
    # numpy.savetxt(store_path + '/input_layer_w.csv', input_layer_weights, delimiter=",")
    # numpy.savetxt(store_path + '/input_layer_b.csv', input_layer_bias, delimiter=",")
    # numpy.savetxt(store_path + '/hidden0_layer_w.csv', hidden0_layer_weights, delimiter=",")
    # numpy.savetxt(store_path + '/hidden0_layer_b.csv', hidden0_layer_bias, delimiter=",")
    # numpy.savetxt(store_path + '/hidden1_layer_w.csv', hidden1_layer_weights, delimiter=",")
    # numpy.savetxt(store_path + '/hidden1_layer_b.csv', hidden1_layer_bias, delimiter=",")

    store_weights(checkpoint_path,store_path)

    #print(input_layer_weights)
    # print(policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.0.weight"].shape)
    # print(policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.0.bias"].shape)
    # print(policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.2.weight"].shape)
    # print(policy.get_state()["weights"]["encoder.actor_encoder.net.mlp.2.bias"].shape)
    # print(policy.get_state()["weights"]["pi.net.mlp.0.weight"].shape)
    # print(policy.get_state()["weights"]["pi.net.mlp.0.bias"].shape)
    # print(wghts)
    #my_data = genfromtxt(store_path+ '/input_layer_w.csv', delimiter=',')
    #print(my_data)
