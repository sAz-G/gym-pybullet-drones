from typing import Optional
import os

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import numpy as np
import copy

from gymnasium import spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CustomBaseAviary import CustomBaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType

# initial_positions =  np.array([
#     [1.0, -2.4492935982947064e-16, 1.0],
#     [0.8090169943749476, 0.9877852522924729, 1.0],
#     [0.30901699437494773, 1.69510565162951535, 1.0],
#     [-0.3090169943749471, 1.69510565162951536, 1.0],
#     [-0.8090169943749472, 0.9877852522924734, 1.0],
#     [-1.0, 1.2246467991473532e-16, 1.0],
#     [-0.8090169943749475, -0.987785252292473, 1.0],
#     [-0.30901699437494756, -1.69510565162951535, 1.0],
#     [0.30901699437494723, -1.69510565162951536, 1.0],
#     [0.8090169943749473, -0.9877852522924734, 1.0]
#     ])
#
# set_of_targets = np.array([
#     [1.0, -2.4492935982947064e-16, 1.0],
#     [0.8090169943749476, 0.9877852522924729, 1.0],
#     [0.30901699437494773, 1.69510565162951535, 1.0],
#     [-0.3090169943749471, 1.69510565162951536, 1.0],
#     [-0.8090169943749472, 0.9877852522924734, 1.0],
#     [-1.0, 1.2246467991473532e-16, 1.0],
#     [-0.8090169943749475, -0.987785252292473, 1.0],
#     [-0.30901699437494756, -1.69510565162951535, 1.0],
#     [0.30901699437494723, -1.69510565162951536, 1.0],
#     [0.8090169943749473, -0.9877852522924734, 1.0]
#     ])


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
#
#
# set_of_obstacles = np.array([
#                                 [1. ,  1., 1.],
#                                 [-1., -1., 1 ],
#                                 [1. , -1., 1.],
#                                 [-1.,  1., 1 ],
#                                 ])


initial_positions =  np.array([
    [2.0, 0, 1.0],
    [0, 2.0, 1.0]
    ])

set_of_targets = np.array([
    [1.0,0, 1.0],
    [.0, 1.0,1.0]
    ])



class RandomSetsGenerator():
    def __init__(self,
                 bounds = np.array([[-4.,4.],[-4., 4.],[0.,4.]]),
                 max_speedx = 3,
                 max_speedy = 3,
                 max_speedz = 3,
                 set_of_pstns=None,
                 set_of_obstcls=None,
                 set_of_trgts=None,
                 N_q = 10, N_o = 5):
        self.postns     = set_of_pstns
        self.obs_pstns  = set_of_obstcls
        self.targets    = set_of_trgts
        self.N_o        = N_o
        self.N_q        = N_q
        self.seed_seqx  = 0
        self.seed_seqy  = 4000
        self.seed_seqz  = 8000
        self.seed_seqvx = 12000
        self.seed_seqvy = 16000
        self.seed_seqvz = 20000
        self.bounds     = bounds
        self.max_speedx = max_speedx
        self.max_speedy = max_speedy
        self.max_speedz = max_speedz

        self.rng_x      = np.random.default_rng(self.seed_seqx)
        self.rng_y      = np.random.default_rng(self.seed_seqy)
        self.rng_z      = np.random.default_rng(self.seed_seqz)
        self.rng_vx     = np.random.default_rng(self.seed_seqvx)
        self.rng_vy     = np.random.default_rng(self.seed_seqvy)
        self.rng_vz     = np.random.default_rng(self.seed_seqvz)


    def random_pos_from_bound(self,k):
        x_max = max(self.bounds[0])
        y_max = max(self.bounds[0])
        z_max = max(self.bounds[0])

        x_min = min(self.bounds[0])
        y_min = min(self.bounds[0])
        z_min = min(self.bounds[0])

        x = self.rng_x.random(k)*(x_max-x_min) - x_max
        y = self.rng_y.random(k)*(y_max-y_min) - y_max
        z = self.rng_z.random(k)*(z_max-z_min) - z_max

        return np.array([x,y,z]).reshape((k,3))

    def random_vel_from_bound(self,k):
        x_max = self.max_speedx*2
        y_max = self.max_speedy*2
        z_max = self.max_speedz*2

        x_min = 0
        y_min = 0
        z_min = 0

        x = self.rng_vx.random(k) * (x_max - x_min) - x_max*.5
        y = self.rng_vy.random(k) * (y_max - y_min) - y_max*.5
        z = self.rng_vz.random(k) * (z_max - z_min) - z_max*.5

        return np.array([x,y,z]).reshape((k,3))

    def random_quad_pos(self,q):
        pass

    def random_obst_pos(self,o):
        pass

    def random_obst_vel(self,o):
        pass

    def reset_seed(self):
        self.seed_seqx = self.seed_seqx + 1
        self.seed_seqy = self.seed_seqy + 1
        self.seed_seqz = self.seed_seqz + 1
        self.seed_seqvx = self.seed_seqvx + 1
        self.seed_seqvy = self.seed_seqvy + 1
        self.seed_seqvz = self.seed_seqvz + 1
        self.rng_x = np.random.default_rng(self.seed_seqx)
        self.rng_y = np.random.default_rng(self.seed_seqy)
        self.rng_z = np.random.default_rng(self.seed_seqz)
        self.rng_vx = np.random.default_rng(self.seed_seqvx)
        self.rng_vy = np.random.default_rng(self.seed_seqvy)
        self.rng_vz = np.random.default_rng(self.seed_seqvz)



class CustomRl3(CustomBaseAviary, MultiAgentEnv):
    # add drone id here in case you continue with this implementation. Do it as if it is an agent is learning
    IS_GUI =False
    def __init__(self,
                 conf             = None,
                 num_drones: int  = 2,
                 k_neighbours     = 1,
                 N_o              = 0,
                 set_of_targets   = set_of_targets,
                 obser_size       = 13,
                 act_size         = 4,
                 set_of_positions = initial_positions, # consider removing the global variable and use the rendering environment to obtain this information
                 set_of_obs_pos   = None,
                 max_vel          = 30,
                 xyz_dim          = 4,
                 act_type         = ActionType.VEL,
                 gui              = False,
<<<<<<< HEAD
                 episode_len_step = 10**3
=======
                 episode_len_step = 10**4
>>>>>>> 65be40578a91dea13835a72955576f350bbfe36a
                 ):

        #print("I AM CONFIG", conf)
        self.set_of_positions    = set_of_positions[0:num_drones,:]
        self.set_of_targets      = set_of_targets[0:num_drones,:]

        # parameters for the base aviary
        drone_model: DroneModel = DroneModel.CF2X
        num_drones: int = num_drones
        neighbourhood_radius: float = np.inf
        initial_xyzs = copy.deepcopy(self.set_of_positions)
        initial_rpys = None
        physics: Physics = Physics.PYB
        pyb_freq:  int = 240
        ctrl_freq: int = 240


        record              = False
        obstacles           = False
        user_debug_gui      = False
        vision_attributes   = False
        output_folder       = 'results'


        self.ACT_TYPE         = act_type
        self.N_q              = num_drones
        self.N_o              = N_o
        self.OBS_SIZE         = obser_size
        self.ACT_SIZE         = act_size
        self.set_of_targets   = copy.deepcopy(self.set_of_targets)
        self.set_of_quad_pos  = self.set_of_positions
        self.set_of_quad_vel  = np.zeros((self.N_q, 3))
        self.set_of_obs_pos   = set_of_obs_pos
        self.set_of_obs_vel   = np.zeros((self.N_o, 3))
        self.k_neighbours     = k_neighbours
        self.MAX_LIN_V        = max_vel
        self.MAX_DISTANCE     = np.sqrt(3) * xyz_dim
        self.MAX_BEARING      = np.pi
        self.col_radius       = 7
        self.MAX_XYZ          = xyz_dim
        self.terminateds      = set()
        self.truncateds       = set()
        self.NUM_DRONES       = num_drones
        self._agent_ids       = set(range(self.NUM_DRONES))
        self.EPISODE_LEN_SEC  = 1500
        self.EPISODE_LEN_STEP = episode_len_step
        self.observation_dict = {k: None for k in range(self.N_q)}
        self.action_dict      = {k: 0 for k in range(self.N_q)}


        if act_type in [ActionType.VEL]:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print(
                    "[ERROR] in CollAvoidAviary.__init()__, no controller is available for the specified drone_model")

        # initialize superclass
        CustomBaseAviary.__init__(self,drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,  # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=user_debug_gui,  # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         output_folder = output_folder
                         )

        self.SPEED_LIMIT = 0.08 * self.MAX_SPEED_KMH * (1000 / 3600)

        MultiAgentEnv.__init__(self)
        #self._addObstacles()


    ######################## GETTERS ######################################
    def get_observation(self, q):
        self.update_information()
        obs_vec_coll = self.get_avoidance_observaion(q)  # observation related to collision avoidance
        obs_vec = obs_vec_coll
        return obs_vec

    def get_avoidance_observaion(self, q):
        """
          get observatoin related to collision avoidance

          IDEAS:
            - consider observing also previous observed states and not only currently observed
            - start with only the current state
            - observation: velocities
          QUESTIONS:
            - how many k neighbours to observe?
            - what happens, in case less than k neighbours are in the collision radius (sensing range)?
            - why not considering also acceleration?
            - do we have to add noise to current observations? (gaußian as in the paper)
            - Does it make sense to use adjacency matrix?
            - How does the size of the input layer affects the learning od the agent? are there to big input layer?
            - Regards obstacles, how to model their position (or shape)?
          TODOS:
            - look for proper acceleration model
        """
        # boundary parameters
        max_lin_v =      self.MAX_LIN_V
        max_dist =       self.MAX_DISTANCE
        max_bearing =    self.MAX_BEARING


        # position related observations
        target_pos = self.get_target_position(q)
        own_pos = self.get_quad_pos(q)
        own_vel = self.get_quad_vel(q)

        target_distance = np.linalg.norm(target_pos - own_pos)

        target_to_own_y = target_pos[1] - own_pos[1]
        target_to_own_x = target_pos[0] - own_pos[0]
        bearing2target = np.arctan2(target_to_own_y, target_to_own_x)

        peer_pos, peer_vels, k_nearest = self.get_k_nearest_pos(q)

        k_neighbours = self.k_neighbours
        rel_vel = np.zeros((k_neighbours,3)) + 2.0 * max_lin_v
        rel_dist = np.zeros(k_neighbours) + 2.0 * max_dist
        bearing2peers = np.zeros(k_neighbours) + 2.0 * max_bearing

        if k_nearest.size != 0:

            rel_vel[0:len(k_nearest),:] = (peer_vels[0:len(k_nearest),:] - own_vel)
            rel_pos = (peer_pos[0:len(k_nearest),:] - own_pos)
            rel_dist[0:len(k_nearest)] = np.linalg.norm(rel_pos)

            peers_to_own_y = peer_pos[0:len(k_nearest),1] - own_pos[1]
            peers_to_own_x = peer_pos[0:len(k_nearest),0] - own_pos[0]

            bearing2peers[0:len(k_nearest)] = np.arctan2(peers_to_own_y, peers_to_own_x)

        avoidance_observation = np.hstack([own_pos,
                                           own_vel,
                                           rel_vel.reshape((rel_vel.size,)),
                                           rel_dist,
                                           target_distance,
                                           bearing2target,
                                           bearing2peers,
                                           ])

        return avoidance_observation

    def get_positions_set(self):
        return self.set_of_quad_pos

    def get_velocities_set(self):
        return self.set_of_quad_vel

    def get_position_obstacles(self):
        "q is the observing quadcopter"
        obstacles_pos = np.zeros((self.N_o, 3))
        for o in range(self.N_o):
            obstacles_pos[o, :] = self.get_position_obstacle(o)
        return obstacles_pos.reshape(3 * self.N_o, )

    def get_position_obstacle(self, o):
        return self.set_of_obs_pos[o, :]

    def get_target_position(self, q):
        return self.set_of_targets[q, :]

    def get_k_nearest_pos(self, q):
        "q is the observing quadcopter, which one to take in case more than k neighbours are in collision range"
        max_p_xyz = self.MAX_XYZ
        max_lin_v = self.MAX_LIN_V

        peer_poses = np.zeros(
            (self.k_neighbours,3)) + 2.0 * max_p_xyz  # 2 is the dummy value in case less than k agents are observed
        peer_vels = np.zeros(
            (self.k_neighbours,3)) + 2.0 * max_lin_v  # 2 is the dummy value in case less than k agents are observed

        k_idxs = np.array([None] * self.k_neighbours)  # get idxs of agents, None in case less agents are observed

        k_observed = 0
        for k in range(self.N_q):
            if k == q:
                continue
            elif np.linalg.norm(self.get_quad_pos(k) - self.get_quad_pos(q)) <= self.col_radius:
                peer_poses[k_observed % self.k_neighbours,:] = self.get_quad_pos(k)  # must change the modulo operator
                peer_vels[k_observed % self.k_neighbours,:] = self.get_quad_vel(k)  # must change the modulo operator

                k_idxs[k_observed % self.k_neighbours] = k
                k_observed = k_observed + 1

        if k_observed > self.k_neighbours:
            k_observed = self.k_neighbours

        K_IDXS = np.array(k_idxs[0:k_observed], dtype='int')
        PEER_POSES = peer_poses
        PEER_VELS = peer_vels

        return PEER_POSES, PEER_VELS, K_IDXS

    def get_quad_pos(self, q):
        return self.set_of_quad_pos[q,:]

    def get_quad_vel(self, q):
        return self.set_of_quad_vel[q,:]

    ######################## SETTERS ######################################
    def set_quad_pos(self, q, pos):
        self.set_of_quad_pos[q,:] = pos

    def set_quad_vel(self, q, vel):
        self.set_of_quad_vel[q,:] = vel

    def set_obs_pos(self, q, pos):
        self.set_of_obs_pos[q,:] = pos

    def set_obs_vel(self, q, vel):
        self.set_of_obs_vel[q,:] = vel

    ######################## HELPERS ######################################
    def update_information(self):
        for q in range(0, self.N_q):
            state = self._getDroneStateVector(q)
            pos = state[0:3]
            vel = state[10:13]
            self.set_quad_pos(q, pos)
            self.set_quad_vel(q, vel)

    ######################## REINFORCEMENT LEARNING #######################
    def _observationSpace(self):
        """
        """
        own_pos_min_xyz = -1.
        own_pos_max_xyz = 1.

        own_vel_min_xyz = -1.
        own_vel_max_xyz = 1.

        vel_p_min_xyz = -1.
        vel_p_max_xyz = 2.

        dist_t_min = 0.
        dist_t_max = 1.

        dist_p_min = 0.
        dist_p_max = 2.

        bearing_t_min = 0.
        bearing_t_max = 1.

        bearing_p_min = 0.
        bearing_p_max = 2.

        low_bound = np.array(
            [own_pos_min_xyz, own_pos_min_xyz,
             own_pos_min_xyz, own_vel_min_xyz,
             own_vel_min_xyz, own_vel_min_xyz,
             vel_p_min_xyz, vel_p_min_xyz,
             vel_p_min_xyz, dist_p_min,
             dist_t_min, bearing_t_min, bearing_p_min] , dtype=np.float32)

        high_bound = np.array(
            [own_pos_max_xyz, own_pos_max_xyz, own_pos_max_xyz,
             own_vel_max_xyz, own_vel_max_xyz, own_vel_max_xyz,
             vel_p_max_xyz, vel_p_max_xyz, vel_p_max_xyz,
             dist_p_max, dist_t_max, bearing_t_max,
             bearing_p_max], dtype=np.float32)

        return spaces.Box(low=low_bound,
                                          high=high_bound,
                                          dtype=np.float32
                                          )

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):

        self.terminateds = set()
        self.truncateds = set()

        self.set_of_quad_pos = self.set_of_positions
        self.set_of_quad_vel = np.zeros((self.N_q, 3))
        self.observation_dict = {k: None for k in range(self.N_q)}
        self.action_dict = {k: 0 for k in range(self.N_q)}

        obs, infos =  super().reset(seed=seed, options=options)

        return obs, infos

    #@override(MultiAgentEnv)
    def step(
            self, action
    ):
        # print()
        # print("Information before next step: ")
        # print()
        # info = self.get_info()
        #
        # print()
        # print('position',info[0]['pos'])
        # print()
        #
        # print('velocity',info[0]['vel'])
        # print()
        #
        # print('action',  info[0]['action'])
        # print()
        #
        # print('distance',info[0]['dist_targ'])
        # print()
        #
        # print('observation',info[0]['observation'])
        print(action)
        action_vel ={}
        for k,v in enumerate(action.values()):
             action_vel[k] = np.random.normal(v[0:4], v[3:7])
       # print("I AM ACTION SPACE",action)
        obs, rewards, terminateds, truncateds, infos = super().step(action_vel)
        #print("obs {}".format(obs) )
        #print("obs shape{}".format(obs[0].shape) )
        # print("rewards {}".format(rewards) )
        # print("terminateds {}".format(terminateds))
        # print("truncateds {}".format(truncateds))
        # print("infos {}".format(infos))
        return obs, rewards, terminateds, truncateds, infos

    def _computeInfo(self):
        info_dict = {i: {} for i in range(self.N_q)}
        return info_dict

    def get_info(self):
        info_dict = {i: {} for i in range(self.N_q)}
        for q in range(self.N_q):
            pos_q = self.get_quad_pos(q)
            vel_q = self.get_quad_vel(q)
            pos_q = pos_q[0:3]
            targ_q = self.get_target_position(q)
            dist_q = np.linalg.norm(pos_q - targ_q)

            obs = self.observation_dict[q]
            obs[0] = obs[0]*self.MAX_XYZ
            obs[1] = obs[1]*self.MAX_XYZ
            obs[2] = obs[2]*self.MAX_XYZ

            obs[3] = obs[3] * self.MAX_LIN_V
            obs[4] = obs[4] * self.MAX_LIN_V
            obs[5] = obs[5] * self.MAX_LIN_V

            info_dict[q] = {"pos" : pos_q,
                            "vel" : vel_q,
                            "dist_targ" : dist_q,
                            "action"   : self.last_action,
                            "observation" : self.observation_dict[q]
                            }
        return info_dict

    def is_out_of_bounds(self,pos):
        if np.abs(pos[0]) > self.MAX_XYZ:
            print("OUT OF BOUNDS")
            return True
        elif np.abs(pos[1]) > self.MAX_XYZ:
            print("OUT OF BOUNDS")
            return  True
        elif np.abs(pos[2]) > self.MAX_XYZ:
            print("OUT OF BOUNDS")
            return True
        else:
            return False

    def _computeTruncated(self):
        all_val = False
        truncated = {i: False for i in range(self.NUM_DRONES)}

        for q in range(self.NUM_DRONES):
            pos_q        = self.get_quad_pos(q)[0:3]

            trunc = False
            if np.abs(pos_q[0]) > self.MAX_XYZ:
                trunc = True
            elif np.abs(pos_q[1]) > self.MAX_XYZ:
                trunc = True
            elif np.abs(pos_q[2]) > self.MAX_XYZ:
                trunc = True

            trunc = trunc or (self.step_counter > self.EPISODE_LEN_STEP)

            all_val      = all_val and trunc

        truncated = {i: all_val for i in range(self.NUM_DRONES)}
        truncated["__all__"] = all_val
        return truncated


    def _computeTerminated(self):
<<<<<<< HEAD
        arrived_dist = .33
        bool_val = (self.step_counter > self.EPISODE_LEN_STEP)
=======
        arrived_dist = 0.1
        bool_val = False
>>>>>>> 65be40578a91dea13835a72955576f350bbfe36a
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        all_val = True

        for q in range(self.NUM_DRONES):
            pos_q   = self.get_quad_pos(q)[0:3]
            targ_q  = self.get_target_position(q)
            dist_q  = np.linalg.norm(pos_q-targ_q)
            #done[q] = (dist_q <= arrived_dist)

            all_val = (all_val and (dist_q <= arrived_dist) ) #done[q]

        if all_val:
            print("ALL VAL IS ON")

        done = {i: all_val for i in range(self.NUM_DRONES)}
        done["__all__"] = all_val
        return done

    def _computeObs(self):
        obs_final = {}
        for i in range(self.NUM_DRONES):
            obs = self._clipAndNormalizeState(self.get_observation(i))  # observation of positions to avoid
            obs_final[i] = np.array(obs, dtype=np.float32)
        self.observation_dict = obs_final
        return obs_final

    def _computeReward(self):
        swarm_size = self.NUM_DRONES
        rewards = {}
        states = np.array([self._getDroneStateVector(i) for i in range(swarm_size)])

        for k in range(0, swarm_size):
            other_peers = np.array([u for u in range(0, swarm_size) if u != k], dtype='int')

            own_pos = states[k, 0:3]
            other_pos = states[other_peers, 0:3]
            own_target = self.set_of_targets[k, 0:3]
            own_vel = states[k, 10:13]#self.get_quad_vel(k)

            dist = (own_pos - other_pos) ** 2
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)

            N_k = len(dist[dist < self.col_radius])
            target_vec = own_target - own_pos

            if np.linalg.norm(own_vel)*np.linalg.norm(target_vec) < .0000001:
                abs = .0000001
            else:
                abs = np.linalg.norm(own_vel)*np.linalg.norm(target_vec)
            dot_prod_vp = np.dot(own_vel, target_vec)/abs

<<<<<<< HEAD
            rewards[k] = dot_prod_vp - 1.0 * N_k
=======
            rewards[k] = 100000.0*dot_prod_vp # - 1.0 * N_k

            if np.abs(own_pos[0]) > self.MAX_XYZ:
                #rewards[k] += -10**12
                rewards[k] += np.exp(np.abs(own_pos[0]) + self.MAX_XYZ)
            elif np.abs(own_pos[1]) > self.MAX_XYZ:
                #rewards[k] += -10**12
                rewards[k] += np.exp(np.abs(own_pos[1]) + self.MAX_XYZ)
            elif np.abs(own_pos[2]) > self.MAX_XYZ:
                rewards[k] += np.exp(np.abs(own_pos[2]) + self.MAX_XYZ)


>>>>>>> 65be40578a91dea13835a72955576f350bbfe36a
        return rewards


    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        # if self.ACT_TYPE in [ActionType.VEL]:
        #     size = 4
        # else:
        #     print("[ERROR] in BaseMultiagentAviary._actionSpace()")
        #     exit()
        # act_lower_bound = np.array([-1 * np.ones(size)])
        # act_upper_bound = np.array([+1 * np.ones(size)])
        # return spaces.Box(low=-1 * np.ones(size),
        #                            high=np.ones(size),
        #                            dtype=np.float32
        #                            )

        if self.ACT_TYPE in [ActionType.VEL]:
            size = 8
        else:
            print("[ERROR] in BaseMultiagentAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1, -1, -1, 0, 0, 0,0,0])
        act_upper_bound = np.array([1,  1,   1, 1, 1, 1,1,1])
        return spaces.Box(low=act_lower_bound,
                          high=act_upper_bound,
                          dtype=np.float32
                          )


    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent
        RPMs, desired thrust and torques, the next target position to reach
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : dict[str, ndarray]
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        rpm = np.zeros((self.NUM_DRONES,4))
        for k, v in action.items():
            if self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(int(k))

                if np.linalg.norm(v[0:3]) != 0:
                    v_unit_vector = v[0:3] / np.linalg.norm(v[0:3])
                else:
                    v_unit_vector = np.zeros(3)

                temp, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(v[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[int(k),:] = temp
            else:
                print("[ERROR] in BaseMultiagentAviary._preprocessAction()")
                exit()
        return rpm


    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

               Parameters
               ----------
               state : ndarray
                   (20,)-shaped array of floats containing the non-normalized state of a single drone.

               Returns
               -------
               ndarray
                   (20,)-shaped array of floats containing the normalized state of a single drone.

               """

        # maximal range values
        MAX_LIN_VEL     = self.MAX_LIN_V # maximal range linear velocity
        MAX_XYZ         = self.MAX_XYZ  # maximal range position (4x4x4)
        MAX_OBST_DIST   = np.sqrt(3 * MAX_XYZ ** 2)
        MAX_BEARING     = self.MAX_BEARING # max bearing to target or peer

       # MAX_PITCH_ROLL  = np.pi  # Full range

        clipped_pos_xyz              = np.clip(state[0:3],      -MAX_XYZ, MAX_XYZ)                      # own position xy
        clipped_vel_xyz              = np.clip(state[3:6],      -MAX_LIN_VEL, MAX_LIN_VEL)    # own velocity xy
        clipped_peer_rel_veloc       = np.clip(state[6:9],      -2.0*MAX_LIN_VEL, 2.0*MAX_LIN_VEL)
        clipped_distance            = np.clip(state[9],     0., 2.0*MAX_OBST_DIST)
        clipped_distance_t            = np.clip(state[10],     0., MAX_OBST_DIST)
        clipped_bearing_t              = np.clip(state[11],    0. , MAX_BEARING)
        clipped_bearing              = np.clip(state[12],    0. , 2.0*MAX_BEARING)

        # normalize the values
        normalized_pos_xyz              = clipped_pos_xyz  / MAX_XYZ
        normalized_vel_xyz              = clipped_vel_xyz  / MAX_LIN_VEL
        normalized_peer_rel_veloc       = clipped_peer_rel_veloc     / MAX_LIN_VEL
        normalized_distance            = clipped_distance          / MAX_OBST_DIST
        normalized_distance_t            = clipped_distance_t          / MAX_OBST_DIST
        normalized_bearing_t              = clipped_bearing_t            / MAX_BEARING
        normalized_bearing              = clipped_bearing            / MAX_BEARING

        norm_and_clipped = np.hstack([normalized_pos_xyz,
                                      normalized_vel_xyz,
                                      normalized_peer_rel_veloc,
                                      normalized_distance,
                                      normalized_distance_t,
                                      normalized_bearing_t,
                                      normalized_bearing
                                      ])#.reshape(20, )

        return norm_and_clipped

    # def _addObstacles(self):
    #     """Add obstacles to the environment.
    #
    #            These obstacles are loaded from standard URDF files included in Bullet.
    #
    #            """
    #     p.loadURDF("samurai.urdf",
    #                physicsClientId=self.CLIENT
    #                )
    #     p.loadURDF("duck_vhacd.urdf",
    #                [-.5, -.5, .05],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )
    #     p.loadURDF("cube_no_rotation.urdf",
    #                [-.5, -2.5, .5],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )
    #     p.loadURDF("sphere2.urdf",
    #                [0, 2, .5],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT
    #                )

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


if __name__ == '__main__':
    rdm = RandomSetsGenerator()

    print(rdm.random_pos_from_bound(3))
    print(rdm.random_pos_from_bound(3))
    print(rdm.random_pos_from_bound(3))

    print()
    print()
    print(rdm.random_vel_from_bound(3))
