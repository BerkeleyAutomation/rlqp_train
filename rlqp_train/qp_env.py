import numpy as np
from rlqp import RLQP
# from osqp_benchmarks.problem_classes.random_qp import RandomQPExample
# from osqp_benchmarks.problem_classes.portfolio import PortfolioExample
# from osqp_benchmarks.problem_classes.lasso import LassoExample
# from osqp_benchmarks.problem_classes.svm import SVMExample
# from osqp_benchmarks.problem_classes.control import ControlExample
from rlqp_benchmarks.benchmark_problems.example import EXAMPLES_MAP
import logging

log = logging.getLogger("qp_env")

Y_MIN, Y_MAX = -1e6, 1e6
AX_MIN, AX_MAX = -1e6, 1e6

class Box:
    def __init__(self, low, high, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = self.low.shape
        self.size = self.low.size
        self.dtype = dtype

    def sample(self, rng):
        return rng.uniform(low=self.low, high=self.high, size=self.shape).astype(self.dtype)

class ExpBox:
    def __init__(self, low, high, dtype=np.float32):
        self.low_exp = np.log10(low)
        self.high_exp = np.log10(high)
        self.low = low
        self.high = high
        self.shape = self.low.shape
        self.size = self.low.size
        self.dtype = dtype

    def sample(self, rng):
        return 10**rng.uniform(low=self.low_exp, high=self.high_exp, size=self.shape).astype(self.dtype)
    

class QPEpisode:
    def __init__(self, P, q, A, l, u, eps, step_reward, iterations_per_step):
        initial_rho = 0.1 # TODO: configurable.
        settings = {
            'verbose': False,
            'adaptive_rho': False,
            'eps_rel': eps,
            'eps_abs': eps,
            'rho': initial_rho,
            'polish': False,
            'max_iter': iterations_per_step,
            'check_termination': iterations_per_step,
            'eps_prim_inf': 1e-15, # disable infeasibility check
            'eps_dual_inf': 1e-15, # disable infeasibility check
        }
        self.step_reward = step_reward
        self.solver = RLQP()
        self.solver.setup(P, q, A, l, u, **settings)
        x = np.zeros(q.shape)
        y = np.zeros(u.shape)
        self.solver.warm_start(x, y)
        self.info = self.solver.solve().info # solve one to populate info
        self.lower_bound = self.solver._model.lower_bound
        self.upper_bound = self.solver._model.upper_bound

    def get_obs(self):
        return np.stack([
            self.solver._model.Ax,
            self.solver._model.y,
            self.solver._model.z,
            
            self.solver._model.lower_bound,
            self.solver._model.upper_bound,
            self.solver._model.rho_vec
        ], axis=-1)
    
    # def get_obs_orig(self):
    #     lo = self.solver._model.z - self.lower_bound
    #     hi = self.upper_bound - self.solver._model.z
    #     return np.stack([
    #         np.log10(np.clip(np.minimum(lo, hi), 1e-8, 1e6)),
    #         np.clip(self.solver._model.y, Y_MIN, Y_MAX),
    #         np.clip(self.solver._model.z - self.solver._model.Ax, AX_MIN, AX_MAX),
    #         np.log10(self.solver._model.rho_vec)
    #     ], axis=-1)

    def done(self):
        return self.info.status == 'solved'

    def step(self, action):
        """Sets the rho_vec on the QP, and then solves the QP for a number of iterations.
        action is [m, 1] values for the rho vector.
        """
        self.solver._model.rho_vec = action # 10**action
        self.info = self.solver.solve().info
        done = (self.info.status == "solved")
        reward = self.step_reward * (not done)
        return self.get_obs(), reward, done, {}

class BenchmarkGen:
    def __init__(self, problem_class, min_dim, max_dim):
        self.problem_class = problem_class
        self.min_dim = min_dim
        self.max_dim = max_dim

    def __call__(self, rng, eps, step_reward, iterations_per_step):
        prob_dim = rng.integers(self.min_dim, self.max_dim, endpoint=True)
        log.debug(f"Generating QP {self.problem_class.__name__}, dim={prob_dim}")
        qp = self.problem_class(prob_dim, rng=rng, create_cvxpy_problem=False).qp_problem
        return QPEpisode(qp['P'], qp['q'], qp['A'], qp['l'], qp['u'],
                             eps, step_reward, iterations_per_step)
    
class QPEnv:
    def __init__(self, eps, step_reward, iterations_per_step):
        self.eps = eps
        self.step_reward = step_reward
        self.iterations_per_step = iterations_per_step
        self.problems = []
        # self.problems = [
        #     BenchmarkGen(RandomQPExample, 10, 2000),
        #     BenchmarkGen(PortfolioExample, 5, 150),
        #     BenchmarkGen(LassoExample, 10, 200),
        #     BenchmarkGen(SVMExample, 10, 200),
        #     BenchmarkGen(ControlExample, 10, 100)
        # ]
        self.observation_space = Box(
            # Ax, y, z, l, u, rho
            low = np.array([ -1e6, -1e6, -1e6, -1e6, -1e6, 1e-8 ], dtype=np.float32),
            high= np.array([  1e6,  1e6,  1e6,  1e6,  1e6, 1e+6 ], dtype=np.float32))
            #low= np.array([ -8.0, Y_MIN, AX_MIN, -6.0], dtype=np.float32),
            #high=np.array([  6.0, Y_MAX, AX_MAX,  6.0], dtype=np.float32))
        self.action_space = ExpBox(
            low=np.array([1e-6], dtype=np.float32),
            high=np.array([1e6], dtype=np.float32))

    def add_benchmark_problem_class(self, name, min_dim, max_dim):
        log.info(f"Adding {name} problem class with dimension range=[{min_dim}, {max_dim}]")
        problem_class = EXAMPLES_MAP[name]
        self.problems.append(
            BenchmarkGen(problem_class, min_dim, max_dim))

    def _random_episode(self, no, rng):
        # Do not randomize the order of the QPs.  We need an even
        # distribution from them, and some random QPs result in
        # "already solved" more frequently than others, leading to an
        # uneven distribution.
        qp_gen = self.problems[no % len(self.problems)]
        return qp_gen(rng, self.eps, self.step_reward, self.iterations_per_step)
        
    def new_episode(self, no, rng=None):
        if rng is None:
            rng = np.random.default_rng(seed=no)

        episode = self._random_episode(no, rng)
        while episode.done():
            log.debug("  already solved")
            episode = self._random_episode(no, rng)

        return episode
            
