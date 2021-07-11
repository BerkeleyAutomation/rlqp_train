import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import logging
from copy import deepcopy
from rlqp_train.replay_buffer import ReplayBuffer
from rlqp_train.util import mlp, freeze, frozen
from rlqp_train.epoch_logger import EpochLogger
from rlqp_train.rho_vec_input import Mode8

import os

log = logging.getLogger("ddpg")

class ExpTanh(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        assert(min_val.shape == (1,)) # TODO: handle different input shapes better
        assert(max_val.shape == (1,))
        min_exp = np.log10(min_val[0])
        max_exp = np.log10(max_val[0])
        self.scale = (max_exp - min_exp) * 0.5
        self.min_exp = min_exp
        
    def forward(self, x):
        x = (torch.tanh(x) + 1.0) * self.scale + self.min_exp
        return 10.0**x

class Actor(nn.Module):
    def __init__(self, input_encoding, act_dim, hidden_sizes, activation, output_activation): #act_min, act_max):
        super().__init__()
        self.input_encoding = input_encoding() # TODO: remove obs_dim, since Mode will be the input
        layer_sizes = [self.input_encoding.output_dim] + list(hidden_sizes) + [act_dim]
        log.info(f"Actor layers: {layer_sizes}")
        self.mlp = mlp(layer_sizes, activation, output_activation=output_activation) #, input_transform=self.input_mode)

    def forward(self, obs):
        return self.mlp(self.input_encoding(obs))

class Critic(nn.Module):
    def __init__(self, input_encoding, act_dim, hidden_sizes, activation):
        super().__init__()
        self.input_encoding = input_encoding() # TODO: replace obs_dim with this.
        self.mlp = mlp([self.input_encoding.output_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        arg = torch.cat([self.input_encoding(obs), torch.log10(act)], dim=-1)
        q = self.mlp(arg)
        return torch.squeeze(q, -1)

class ActorCritic(nn.Module):
    def __init__(self, input_encoding, act_dim, hidden_sizes, activation, output_activation):
        super().__init__()
        self.pi = Actor(input_encoding, act_dim, hidden_sizes, activation, output_activation)
        self.q = Critic(input_encoding, act_dim, hidden_sizes, activation)

    def act(self, obs):
        return self.pi(obs).detach().numpy()

class DDPG:
    def __init__(self, save_dir, env, hparams):
        self.env = env

        self.epoch_logger = EpochLogger(save_dir=save_dir)
        self.epoch_logger.save_settings(hparams)
        
        obs_dim = env.observation_space.size
        act_dim = env.action_space.size

        # Hyperparameters
        self.hparams = hparams        

        # Seed
        self.rng = np.random.default_rng(hparams.seed) # TODO: seed
        torch.manual_seed(hparams.seed)

        input_encoding = Mode8 # TODO: get from caller.
        output_activation = lambda : ExpTanh(
            env.action_space.low, env.action_space.high)
        
        self.ac = ActorCritic(
            input_encoding,
            act_dim,
            hidden_sizes = hparams.hidden_sizes,
            activation = nn.ReLU,
            output_activation = output_activation)
        
        self.ac_targ = deepcopy(self.ac)
        freeze(self.ac_targ, True)

        self.replay_buffer = ReplayBuffer(os.path.join(save_dir, "replay_buffer"),
                                              obs_dim, act_dim, hparams.replay_size)

        checkpoint = self.epoch_logger.load_checkpoint() or dict(
            pi_lr=hparams.pi_lr,
            q_lr=hparams.q_lr,
            ep_no=0,
            epoch_no=0,
            prev_update=0,
            next_update=hparams.update_after,
            next_epoch=hparams.steps_per_epoch,
            test_no=0)
        
        self.pi_opt = Adam(self.ac.pi.parameters(), lr=checkpoint['pi_lr'])
        self.q_opt = Adam(self.ac.q.parameters(), lr=checkpoint['q_lr'])
        
        self.pi_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.pi_opt, step_size=1, gamma=hparams.lr_decay_rate)
        self.q_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.q_opt, step_size=1, gamma=hparams.lr_decay_rate)

        self.ep_no = checkpoint['ep_no']
        self.epoch_no = checkpoint['epoch_no']
        self.prev_update = checkpoint['prev_update']
        self.next_update = checkpoint['next_update']
        self.next_epoch = checkpoint['next_epoch']
        self.test_no = checkpoint['test_no']

        if self.epoch_no > 0:
            self.ac.load_state_dict(checkpoint['ac'])
            self.ac_targ.load_state_dict(checkpoint['ac_targ'])
            self.pi_opt.load_state_dict(checkpoint['pi_opt'])
            self.q_opt.load_state_dict(checkpoint['q_opt'])
            self.pi_lr_scheduler.load_state_dict(checkpoint['pi_lr_scheduler'])
            self.q_lr_scheduler.load_state_dict(checkpoint['q_lr_scheduler'])

    def compute_q_loss(self, obs, act, rew, ob2, don):
        q = self.ac.q(obs, act)
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(ob2, self.ac_targ.pi(ob2))
            backup = rew + self.hparams.gamma * (1 - don) * q_pi_targ
        loss_q = ((q - backup)**2).mean()
        q_vals = q.detach().numpy()[0]
        return loss_q, q_vals

    def compute_pi_loss(self, obs):
        q_pi = self.ac.q(obs, self.ac.pi(obs))
        return -q_pi.mean()

    def update(self, data):
        self.q_opt.zero_grad()
        loss_q, q_vals = self.compute_q_loss(**data)
        loss_q.backward()
        self.q_opt.step()

        with frozen(self.ac.q):
            self.pi_opt.zero_grad()
            loss_pi = self.compute_pi_loss(data['obs'])
            loss_pi.backward()
            self.pi_opt.step()

        self.epoch_logger.accum(LossQ=loss_q.item(), LossPi=loss_pi.item(), QVals=q_vals.item())

        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.hparams.polyak)
                p_targ.data.add_((1.0 - self.hparams.polyak) * p.data)

    def get_action(self, obs, noise_scale, rng):
        """Uses the policy to compute an action, optionally adding noise."""
        a = self.ac.act(torch.as_tensor(obs, dtype=torch.float32))
        if noise_scale:
            # a += noise_scale * rng.standard_normal(self.env.action_space.size)
            # In prior code we added noise before computing the
            # exponent.  I.e., action = 10**(a + N) Now we're using:
            #   a' = 10**a, so we compute
            # action=10**(a + N) = a' * 10**N
            a *= 10**(noise_scale * rng.standard_normal(self.env.action_space.size))
            
        return np.clip(a, self.env.action_space.low[0],
                          self.env.action_space.high[0])

    def test_agent(self, test_no):
        rng = np.random.default_rng(test_no + int(1e9))
        episode = self.env.new_episode(test_no, rng=rng)
        obs, done, ep_len, ep_ret = episode.get_obs(), False, 0, 0
        while ep_len < self.max_ep_len and not done:
            obs, rew, done, _ = episode.step(self.get_action(obs, 0, rng))
            ep_len += 1
            ep_ret += rew

        return ep_len, ep_ret

    def random_action(self, o, rng):
        """Computes a random action to take.

        This supports both scalar- and vector-based training.  If
        vector-based, and the action is a scalar, it is tiled across to
        all the action vector.
        """
        a = self.env.action_space.sample(rng)
        if not np.isscalar(o) and o.shape[0] != self.env.observation_space.shape[0]:
            a = np.tile(a, (o.shape[0], 1))
            # TODO: sample the vector space, instead of tiling a single action
        return a

    def run_episode(self, ep_no, use_start_actions):
        # To gain some determinism for debugging, and to allow
        # episodes to run on separate threads, each episode gets its
        # own sequentially seeded random number generator.
        rng = np.random.default_rng(ep_no)

        # Generate a new episode
        episode = self.env.new_episode(ep_no, rng=rng)

        obs, done, ep_log, ep_ret = episode.get_obs(), False, [], 0
        
        log.debug(f"obs = {obs.shape}")

        # Step until we've reached the episode length or the episode
        # is done.
        while len(ep_log) < self.hparams.max_ep_len and not done:
            if use_start_actions:
                act = self.random_action(obs, rng)
            else:
                act = self.get_action(obs, self.hparams.act_noise, rng)

            ob2, rew, done, _ = episode.step(act)
            
            ep_ret += rew
            ep_log.append((obs, act, rew, ob2, done))
            obs = ob2

        self.epoch_logger.accum(EpRet=ep_ret, EpLen=len(ep_log))
            
        # if not done:
        #     ep_log = ep_log[:10]

        log.info(f"episode {ep_no} done, len={len(ep_log)}, ret={ep_ret}")

        with self.replay_buffer._lock:
            for o, a, r, o2, done in ep_log:
                self.replay_buffer.store_array(o, a, r, o2, done)

        return ep_ret, len(ep_log)

    def test_episode(self, test_no):
        episode = self.env.new_episode(test_no)
        obs, done, ep_ret, ep_len = episode.get_obs(), False, 0, 0
        while not (done or ep_len == self.hparams.max_ep_len):
            obs, r, done, _ = episode.step(self.get_action(obs, 0, None))
            ep_ret += r
            ep_len += 1
        log.info(f"test episode {test_no} done, len={ep_len}, ret={ep_ret}")
        return ep_ret, ep_len

    def train_epoch(self):
        self.epoch_no += 1
        log.info(f"Starting epoch {self.epoch_no}")
        steps_taken = self.replay_buffer.steps_taken()
        start_index = self.replay_buffer.index()
        while steps_taken < self.next_epoch:
            with torch.no_grad():
                while steps_taken < self.next_epoch and steps_taken < self.next_update:
                    self.ep_no += 1
                    ep_ret, ep_len = self.run_episode(self.ep_no, steps_taken < self.hparams.start_steps)
                    steps_taken = self.replay_buffer.steps_taken()
            
            while steps_taken >= self.next_update:
                for _ in range(self.prev_update, self.next_update):
                    batch = self.replay_buffer.sample_batch(self.rng, self.hparams.batch_size)
                    self.update(data=batch)
                self.prev_update = self.next_update
                self.next_update += self.hparams.update_every

        with torch.no_grad():
            for _ in range(self.hparams.num_test_episodes): # TODO: this is not really a hyper parameter
                self.test_no += 1
                ep_ret, ep_len = self.test_episode(self.test_no)
                self.epoch_logger.accum(TestEpRet=ep_ret, TestEpLen=ep_len)

        self.next_epoch += self.hparams.steps_per_epoch
        
        data_fill = (self.replay_buffer.index() - start_index) / self.replay_buffer.capacity

        self.epoch_logger.epoch(self.epoch_no,
            data_fill=data_fill,
            pi_lr=self.pi_opt.param_groups[0]['lr'],
            q_lr=self.q_opt.param_groups[0]['lr'])            

        self.pi_lr_scheduler.step()
        self.q_lr_scheduler.step()

        self.epoch_logger.save_checkpoint(
            self.epoch_no,
            pi_lr=self.pi_opt.param_groups[0]['lr'],
            q_lr=self.q_opt.param_groups[0]['lr'],
            ep_no=self.ep_no,
            prev_update=self.prev_update,
            next_update=self.next_update,
            next_epoch=self.next_epoch,
            test_no=self.test_no,
            ac=self.ac.state_dict(),
            ac_targ=self.ac_targ.state_dict(),
            pi_opt=self.pi_opt.state_dict(),
            q_opt=self.q_opt.state_dict(),
            pi_lr_scheduler=self.pi_lr_scheduler.state_dict(),
            q_lr_scheduler=self.q_lr_scheduler.state_dict())

    def train(self):
        while self.epoch_no < self.hparams.num_epochs:
            self.train_epoch()
