import os
import logging
import argparse

# https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
class Range:
    def __init__(self, start, end=None):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
    def __contains__(self, item):
        return self.__eq__(item)
    def __iter__(self):
        yield self
    def __repr__(self):
        return 'range [{0}, {1}]'.format(self.start, self.end)

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--save_dir", metavar="DIR", required=True, type=str, help="Directory to save/load checkpoints")

# Environment related options
parser.add_argument("--qp_env", metavar="ENV:MIN:MAX", type=str, nargs='+',
                        default=["Random QP:10:100", "Portfolio:5:15", "Lasso:10:20", "SVM:10:20", "Control:10:10"],
                        help="The environments to use for training")
parser.add_argument("--qp_iters_per_step", metavar="M", type=int, default=200, choices=Range(1, 10000),
                        help="Number of QP ADMM (internal) iterations per adaptation (default: %(default)s)")
parser.add_argument("--qp_step_reward", metavar="R", type=float, default=-1.0, choices=Range(-1e9, -1e-6), help="Reward for each step (default: %(default)s)")
parser.add_argument("--qp_eps", metavar="EPS", type=float, default=1e-6, choices=Range(1e-9, 1.0), help="Set termination epsilon for QP (default: %(default)s)")

# Run configuration
parser.add_argument("--num_workers", metavar="P", type=int, default=0, choices=Range(0,16384), help="Number of worker processes to use.  Set to 0 to use all available CPU cores.  Using a value greater than 1 requires disables OpenMP threading otherwise the training process will deadlock.  (default: %(default)s)")
parser.add_argument("--num_test_episodes", metavar="T", type=int, default=50, choices=Range(0, 1<<32-1), help="Number of test episodes per epoch (default: %(default)s)")
parser.add_argument("--seed", type=int, default=20210708, help="Random number generator seed (default: %(default)s)")

# Hyperparmeters
parser.add_argument("--replay_size", metavar="SIZE", type=int, default=int(1e6), choices=Range(1000,1<<32-1), help="Replay buffer size (default: %(default)s)")
parser.add_argument("--pi_lr", metavar="LR", type=float, default=1e-3, choices=Range(0, 1), help="Initial learning rate for policy (default: %(default)s)")
parser.add_argument("--q_lr", metavar="LR", type=float, default=1e-3, choices=Range(0, 1), help="Initial learning rate for critic (default: %(default)s)")
parser.add_argument("--lr_decay_rate", metavar="RATE", type=float, default=0.999, choices=Range(0, 1), help="Decay step for learning rates (default: %(default)s)")
parser.add_argument("--steps_per_epoch", metavar="N", type=int, default=2000, choices=Range(0, 1<<32-1), help="Steps per epoch (default: %(default)s)")
parser.add_argument("--hidden_sizes", metavar="H", type=int, nargs='+', default=[128, 128, 128], choices=Range(1,1<<20), help="Hidden layer size(s), (default: %(default)s)")
parser.add_argument("--hidden_activation", type=str, default="ReLU", choices={"ReLU"}, help="Hidden layer activation (default: %(default)s)")
parser.add_argument("--num_epochs", metavar="NUM", type=int, default=25, choices=Range(1, 10000), help="Number of training epochs (default: %(default)s)")
parser.add_argument("--max_ep_len", metavar="LEN", type=int, default=100, choices=Range(1, 10000), help="Maximum episode length (default: %(default)s)")
parser.add_argument("--update_every", metavar="STEP", type=int, default=1000, choices=Range(1, 1<<32-1), help="Frequency of network updates (default: %(default)s)")
parser.add_argument("--batch_size", metavar="SIZE", type=int, default=100, choices=Range(1, 1000000), help="Batch size for updates (default: %(default)s)")
parser.add_argument("--gamma", type=float, default=0.99, choices=Range(0, 1), help="Discount factor (default: %(default)s)")
parser.add_argument("--polyak", type=float, default=0.995, choices=Range(0, 1), help="Polyak rate (default: %(default)s)")
parser.add_argument("--act_noise", metavar="VAR", type=float, default=2.0, choices=Range(0.0, 15.0), help="Action exploration noise (default: %(default)s)")
parser.add_argument("--update_after", metavar="STEP", type=int, default=1000, choices=Range(1, 1<<32-1), help="Initial steps before first update (default: %(default)s)")
parser.add_argument("--start_steps", metavar="STEP", type=int, default=5000, choices=Range(1, 1<<32-1), help="Initial steps before using actor policy (default: %(default)s)")
parser.add_argument("--debug", action='store_true', help="Enable debug-level messages")

# Set up logging,
hparams = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if hparams.debug else logging.INFO)
del hparams.debug

if hparams.num_workers == 0:
    hparams.num_workers = os.cpu_count()

if hparams.num_workers > 1:
    # When num_workers > 1, we fork processes to run the QPs.  This
    # causes OpenMP to deadlock when it is using more than 1 thread.
    # Here we disable OpenMP threading to avoid the issue.
    os.environ["OMP_NUM_THREADS"] = "1"

# Import AFTER possibly setting OMP_NUM_THREADS to 1 to avoid
# inadvertently initializing OpenMP's threading.
from rlqp_train.qp_env import QPEnv
from rlqp_train.ddpg import DDPG

# Set up the training environment
env = QPEnv(
    eps = hparams.qp_eps,
    step_reward = hparams.qp_step_reward,
    iterations_per_step=hparams.qp_iters_per_step)

for e in hparams.qp_env:
    name, min_dim, max_dim = e.split(':')
    env.add_benchmark_problem_class(name, int(min_dim), int(max_dim))

del hparams.qp_env, hparams.qp_eps, hparams.qp_step_reward, hparams.qp_iters_per_step

# Set up the RL algorithm and run it
save_dir = hparams.save_dir
del hparams.save_dir
ddpg = DDPG(save_dir=save_dir, env=env, hparams=hparams)

ddpg.train()

    
                    
                
