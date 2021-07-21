import os
import json
import torch
import argparse
import numpy as np
import rlqp_train.ddpg as ddpg
from rlqp_train.epoch_logger import EpochLogger
from rlqp_train.rho_vec_input import Mode8
import rlqp_train.util as util

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str)
parser.add_argument('--checkpoint', type=int)
parser.add_argument('--traced_output', type=str)
args = parser.parse_args()

epoch_logger = EpochLogger(args.save_dir, read_only=True)
print("Loading settings.json")
with open(os.path.join(args.save_dir, "settings.json")) as f:
    settings = json.load(f)

# TODO: make sure to match with ddpg.py
input_encoding = Mode8
output_activation = lambda : ddpg.ExpTanh(
    np.array([1e-6], dtype=np.float32), np.array([1e6], dtype=np.float32))
    # env.action_space.low, env.action_space.high)

print("Creating ActorCritic")
ac = ddpg.ActorCritic(
    input_encoding,
    act_dim = 1,
    hidden_sizes = settings['hidden_sizes'],
    activation = util.activation_name_map[settings['hidden_activation']],
    output_activation = output_activation)

print("Loading weights")
checkpoint = epoch_logger.load_checkpoint(args.checkpoint)
ac.load_state_dict(checkpoint['ac'])

print("Tracing")
example_input = torch.rand((1,6)) # Ax, y, z, l, u, rho
traced_model = torch.jit.trace(ac.pi, example_input)

if args.traced_output is not None:
    print(f"Writing {args.traced_output}")
    traced_model.save(args.traced_output)

print("Done.")
