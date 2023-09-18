from typing import Optional
import os

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import numpy as np
import copy

from gymnasium import spaces

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CustomBaseAviary import CustomBaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType


initial_positions =  np.array([
    [1.0, -2.4492935982947064e-16, 1.0],
    [0.8090169943749476, 0.9877852522924729, 1.0],
    [0.30901699437494773, 1.69510565162951535, 1.0],
    [-0.3090169943749471, 1.69510565162951536, 1.0],
    [-0.8090169943749472, 0.9877852522924734, 1.0],
    [-1.0, 1.2246467991473532e-16, 1.0],
    [-0.8090169943749475, -0.987785252292473, 1.0],
    [-0.30901699437494756, -1.69510565162951535, 1.0],
    [0.30901699437494723, -1.69510565162951536, 1.0],
    [0.8090169943749473, -0.9877852522924734, 1.0]
    ])

set_of_targets = np.array([
    [1.0, -2.4492935982947064e-16, 1.0],
    [0.8090169943749476, 0.9877852522924729, 1.0],
    [0.30901699437494773, 1.69510565162951535, 1.0],
    [-0.3090169943749471, 1.69510565162951536, 1.0],
    [-0.8090169943749472, 0.9877852522924734, 1.0],
    [-1.0, 1.2246467991473532e-16, 1.0],
    [-0.8090169943749475, -0.987785252292473, 1.0],
    [-0.30901699437494756, -1.69510565162951535, 1.0],
    [0.30901699437494723, -1.69510565162951536, 1.0],
    [0.8090169943749473, -0.9877852522924734, 1.0]
    ])

set_of_obstacles = np.array([
                                [1. ,  1., 1.],
                                [-1., -1., 1 ],
                                [1. , -1., 1.],
                                [-1.,  1., 1 ],
                                ])




class CustomRl3(CustomBaseAviary, MultiAgentEnv):
    # add drone id here in case you continue with this implementation. Do it as if it is an agent is learning 
    def __init__(self,
                 conf             = None,
                 num_drones: int  = 3,
                 k_neighbours     = 1,
                 N_o              = 0,
                 N_q              = 3,
                 set_of_targets   = set_of_targets,
                 obser_size       = 13,
                 act_size         = 4,
                 set_of_positions = initial_positions, # consider removing the global variable and use the rendering environment to obtain this information
                 set_of_obs_pos   = None,
                 max_vel          = 100,
                 xyz_dim          = 4,
                 act_type         = ActionType.VEL,
                 ):

        set_of_positions = set_of_positions[0:num_drones,:]
        set_of_targets = set_of_targets[0:num_drones,:]

        # parameters for the base aviary
        drone_model: DroneModel = DroneModel.CF2X
        num_drones: int = num_drones
        neighbourhood_radius: float = np.inf
        initial_xyzs = copy.deepcopy(set_of_positions)
        initial_rpys = None
        physics: Physics = Physics.PYB
        pyb_freq: int = 240
        ctrl_freq: int = 240
        gui = True
        record = False
        obstacles = False
        user_debug_gui = True
        vision_attributes = False
        output_folder = 'results'


        self.ACT_TYPE = act_type
        # custom reinforcement learning parameters
        self.N_q = N_q
        self.N_o = N_o
        self.OBS_SIZE = obser_size
        self.ACT_SIZE = act_size
        self.set_of_targets = copy.deepcopy(set_of_targets)
        self.set_of_quad_pos = set_of_positions[0:num_drones,:]
        self.set_of_quad_vel = np.zeros((self.N_q, 3))
        self.set_of_obs_pos = set_of_obs_pos
        self.set_of_obs_vel = np.zeros((self.N_o, 3))
        self.k_neighbours = k_neighbours
        #self.observation_space = self._observationSpace()
        #self.action_space = self._actionSpace()
        self.MAX_LIN_V = max_vel
        self.MAX_DISTANCE = np.sqrt(3) * xyz_dim
        self.MAX_BEARING = np.pi
        self.col_radius = 0.6
        self.MAX_XYZ = xyz_dim
        self.terminateds = set()
        self.truncateds = set()
        self.NUM_DRONES = num_drones
        self._agent_ids = set(range(self.NUM_DRONES))
        self.EPISODE_LEN_SEC = 150


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

        self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)

        MultiAgentEnv.__init__(self)







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

        k_neighbours = self.k_neighbours

        # position related observations
        target_pos = self.get_target_position(q)
        own_pos = self.get_quad_pos(q)
        own_vel = self.get_quad_vel(q)

        target_distance = np.linalg.norm(target_pos - own_pos)

        target_to_own_y = target_pos[1] - own_pos[1]
        target_to_own_x = target_pos[0] - own_pos[0]
        bearing2target = np.arctan2(target_to_own_y, target_to_own_x)

        peer_pos, peer_vels, k_nearest = self.get_k_nearest_pos(q)

        rel_vel = np.zeros((3, k_neighbours)) + 2.0 * max_lin_v
        rel_dist = np.zeros(k_neighbours) + 2.0 * max_dist
        bearing2peers = np.zeros(k_neighbours) + 2.0 * max_bearing

        if k_nearest.size != 0:

            rel_vel[:, 0:len(k_nearest)] = (peer_vels[:, 0:len(k_nearest)].T - own_vel).T
            rel_pos = (peer_pos[:, 0:len(k_nearest)].T - own_pos).T
            rel_dist[0:len(k_nearest)] = np.linalg.norm(rel_pos)

            peers_to_own_y = peer_pos[1, 0:len(k_nearest)] - own_pos[1]
            peers_to_own_x = peer_pos[0, 0:len(k_nearest)] - own_pos[0]

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
            (3, self.k_neighbours)) + 2.0 * max_p_xyz  # 2 is the dummy value in case less than k agents are observed
        peer_vels = np.zeros(
            (3, self.k_neighbours)) + 2.0 * max_lin_v  # 2 is the dummy value in case less than k agents are observed

        k_idxs = np.array([None] * self.k_neighbours)  # get idxs of agents, None in case less agents are observed

        k_observed = 0
        for k in range(self.N_q):
            if k == q:
                continue
            elif np.linalg.norm(self.get_quad_pos(k) - self.get_quad_pos(q)) <= self.col_radius:
                peer_poses[:, k_observed % self.k_neighbours] = self.get_quad_pos(k)  # must change the modulo operator
                peer_vels[:, k_observed % self.k_neighbours] = self.get_quad_vel(k)  # must change the modulo operator

                k_idxs[k_observed % self.k_neighbours] = k
                k_observed = k_observed + 1

        # K_IDXS     = np.array(k_idxs[k_idxs != None], dtype='int')

        if k_observed > self.k_neighbours:
            k_observed = self.k_neighbours

        K_IDXS = np.array(k_idxs[0:k_observed], dtype='int')
        PEER_POSES = peer_poses
        PEER_VELS = peer_vels
        # if not (K_IDXS == None).all():
        #     PEER_POSES = peer_poses[0:k_observed, :]
        # else:
        #     PEER_POSES =  peer_poses

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
             vel_p_min_xyz, dist_t_min,
             dist_p_min, bearing_t_min, bearing_p_min] , dtype=np.float32)

        high_bound = np.array(
            [own_pos_max_xyz, own_pos_max_xyz, own_pos_max_xyz,
             own_vel_max_xyz, own_vel_max_xyz, own_vel_max_xyz,
             vel_p_max_xyz, vel_p_max_xyz, vel_p_max_xyz,
             dist_t_max, dist_p_max, bearing_t_max,
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
        obs, infos =  super().reset(seed=seed, options=options)

        return obs, infos

    #@override(MultiAgentEnv)
    def step(
            self, action
    ):
        #for k,v in enumerate(action.values()):
        #    action[k] = np.random.normal(v[0:3], v[3:6])

        obs, rewards, terminateds, truncateds, infos = super().step(action)
        return obs, rewards, terminateds, truncateds, infos

    def _computeTruncated(self):
        bool_val = False
        truncated = {i: bool_val for i in range(self.NUM_DRONES)}
        truncated["__all__"] = False
        return truncated

    def _computeInfo(self):
        return {i: {} for i in range(self.NUM_DRONES)}

    def _computeTerminated(self):
        bool_val = True  if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC else False
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        done["__all__"] = True if True in done.values() else False
        return done

    def _computeObs(self):
        obs_final = {}
        for i in range(self.NUM_DRONES):
            obs = self._clipAndNormalizeState(self.get_observation(i))  # observation of positions to avoid
            obs_final[i] = np.array(obs, dtype=np.float32)

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
            own_vel = self.get_quad_vel(k)

            dist = (own_pos - other_pos) ** 2
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)

            N_k = len(dist[dist < self.col_radius])
            target_vec = own_target - own_pos
            dot_prod_vp = np.dot(own_vel, target_vec)

            rewards[k] = dot_prod_vp - 1.0 * N_k
        return rewards


    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.VEL]:
            size = 4
        else:
            print("[ERROR] in BaseMultiagentAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1 * np.ones(size)])
        act_upper_bound = np.array([+1 * np.ones(size)])
        return spaces.Box(low=-1 * np.ones(size),
                                   high=np.ones(size),
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
        MAX_LIN_VEL     = 30 # maximal range linear velocity
        MAX_XYZ         = 4  # maximal range position (4x4x4)
        MAX_OBST_DIST   = np.sqrt(3 * MAX_XYZ ** 2)
        MAX_BEARING     = np.pi # max bearing to target or peer

       # MAX_PITCH_ROLL  = np.pi  # Full range

        clipped_pos_xyz              = np.clip(state[0:3],      -MAX_XYZ, MAX_XYZ)                      # own position xy
        clipped_vel_xyz              = np.clip(state[3:6],      -MAX_LIN_VEL, MAX_LIN_VEL)    # own velocity xy
        clipped_peer_rel_veloc       = np.clip(state[6:9],      -MAX_LIN_VEL, MAX_LIN_VEL)
        clipped_distances            = np.clip(state[9:11],     0, MAX_OBST_DIST)
        clipped_bearing              = np.clip(state[11:13],    0 , MAX_BEARING)

        # normalize the values
        normalized_pos_xyz              = clipped_pos_xyz  / MAX_XYZ
        normalized_vel_xyz              = clipped_vel_xyz  / MAX_LIN_VEL
        normalized_peer_rel_veloc       = clipped_peer_rel_veloc     / MAX_LIN_VEL
        normalized_distances            = clipped_distances          / MAX_OBST_DIST
        normalized_bearing              = clipped_bearing            / MAX_BEARING


        norm_and_clipped = np.hstack([normalized_pos_xyz,
                                      normalized_vel_xyz,
                                      normalized_peer_rel_veloc,
                                      normalized_distances,
                                      normalized_bearing
                                      ])#.reshape(20, )

        return norm_and_clipped

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

