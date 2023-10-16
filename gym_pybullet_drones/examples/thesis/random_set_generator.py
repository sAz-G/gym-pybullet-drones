import numpy as np

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





