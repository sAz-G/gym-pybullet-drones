"""Simple example of setting up a multi-agent policy mapping.

Control the number of agents and policies via --num-agents and --num-policies.

This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.

Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""
import numpy as np
from numpy import genfromtxt

def load_weights(store_path):
    inp_w = genfromtxt(store_path + '/input_layer_w.csv', delimiter=',')
    inp_b = genfromtxt(store_path + '/input_layer_b.csv', delimiter=',')
    h0_w = genfromtxt(store_path + '/hidden0_layer_w.csv', delimiter=',')
    h0_b = genfromtxt(store_path + '/hidden0_layer_b.csv', delimiter=',')
    h1_w = genfromtxt(store_path + '/hidden1_layer_w.csv', delimiter=',')
    h1_b = genfromtxt(store_path + '/hidden1_layer_b.csv', delimiter=',')

    return inp_w, inp_b, h0_w, h0_b, h1_w, h1_b


if __name__ == "__main__":
    pth =  "C:\\Users\sAz\Documents\GitHub\gym-pybullet-drones\gym_pybullet_drones\examples\\thesis\\nn"
    print()

    inp_w,inp_b,h0_w,h0_b,h1_w,h1_b =  load_weights(pth)


    obs = [0, .5, 1.0,
     0.0, 0.0, 0.0,
     0.0, 0.0, 0.0,
     .8, 1.0, 0.0,
     0.5]

    out = inp_w@obs
    out = out + inp_b

    out = h0_w@out  + h0_b

    out = h1_w@out + h1_b
    print(out)

