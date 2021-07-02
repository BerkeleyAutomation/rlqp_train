import rlqp
import numpy as np

Y_MIN, Y_MAX = -1e6, 1e6
AX_MIN, AX_MAX = -1e6, 1e6

class QPEpisode:
    def __init__(self, P, q, A, l, u, eps, step_reward, iterations_per_step):
        settings = {
            'verbose': False,
            'adaptive_rho': 0,
            'eps_rel': eps,
            'eps_abs': eps,
            'rho': inital_rho,
            'polish': False,
            'max_iter': iterations_per_step,
            'check_termination': iterations_per_step,
            'eps_prim_inf': 1e-15, # disable infeasibility check
            'eps_dual_inf': 1e-15, # disable infeasibility check
        }
        self.solver = rlqp.RLQP()
        self.solver.setup(P, q, A, l, u, **settings)
        x = np.zeros(q.shape)
        y = np.zeros(u.shape)
        self.solver.warm_start(x, y)

    def get_obs(self):
        lo = self.solver._model.z - self.lower_bound
        hi = self.upper_bound - self.solver._model.z
        return np.array([
            np.log10(np.clip(np.minimum(lo, hi), 1e-8, 1e6)),
            np.clip(self.solver._model.y, Y_MIN, Y_MAX),
            np.clip(self.solver._model.z - self.solver._model.Ax, AX_MIN, AX_MAX),
            np.log10(self.solver._model.rho_vec)
        ], dtype=np.float32)

    def step(self, action):
        self.solver._model.rho_vec = 10**action
        self.info = self.solver.solve().info
        done = (self.info.state == "solved")
        reward = self.step_reward * (not done)
        return self.get_obs(), reward, done, {}
