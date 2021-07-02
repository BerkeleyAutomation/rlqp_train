import torch
import torch.nn as nn
from util import mlp, combined_shape, freeze

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.index, self.capacity = 0, size

    def store(self, obs, act, rew, next_obs, done):
        i = self.index % self.capacity
        self.obs_buf[i] = obs
        self.obs2_buf[i] = next_obs
        self.act_buf[i] = act
        self.rew_buf[i] = rew
        self.done_buf[i] = done
        self.index += 1

    def sample_batch(self, rng, batch_size):
        idxs = rng.randint(0, min(self.index, self.capacity), size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.req_buf[idxs],
                     done=self.done_buf[idxs])
        return { k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items() }

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit)
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.pi(obs)

class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64,64),
                     activation=nn.ReLU)
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = QFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = QFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

class TD3:
    HPARAM_SCHEMA = Schema({
        'seed': Use(int),
        'steps_per_epoch': And(Use(int), lambda x: x > 0),
        'replay_size': And(Use(int), lambda x: x > 1000),
        'gamma': And(Use(float), lambda x: 0.0 <= x <= 1.0),
        'polyak': And(Use(float), lambda x: 0.0 < x < 1.0)
        'pi_lr': And(Use(float), lambda x: x > 0.0),
        'q_lr': And(Use(float), lambda x: x > 0.0),
        'batch_size': And(Use(int), lambda x: 10 <= x <= int(1e8)),
        'start_steps': And(Use(int), lambda x: x > 0),
        'update_after': And(Use(int), lambda x: x > 0),
        'update_every': And(Use(int), lambda x: x > 0),
        'act_noise': And(Use(float), lambda x: 0.0 <= x <= 1e9),
        'target_noise': And(Use(float), lambda x: x > 0),
        'noise_clip': And(Use(float), lambda x: x > 0),
        'policy_delay', And(Use(int), lambda x: x > 0),
        'num_test_episodes': And(Use(int), lambda x: x>0),
        'max_ep_len': And(Use(int), lambda x: x > 1),
        })
    def __init__(self, env, replay_size):
        self.env = env

        # TODO: seed

        self.ac = ActorCritic(env.observation_space, env.action_space)
        self.ac_targ = deepcopy(self.ac)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.q_params = itertools.chain(
            self.ac.q1.parameters(),
            self.ac.q2.parameters())

        self.replay_buffer = ReplayBuffer(obs_dim=env.observation_space.shape,
                                              act_dim=env.action_space.shape,
                                              size=replay_size)

        self.pi_optimizer = Adam(self.ac.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q_params, lr=q_lr)

    def compute_loss_q(self, obs, act, rew, obs2, done):
        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2)

            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)

            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)

            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    def compute_loss_pi(self, obs):
        q1_pi = self.ac.q1(obs, self.pi(obs))
        return -q1_pi.mean()

    def update(self, data, timer):
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(*data)
        loss_q.backward()
        self.q_optimizer.step()

        # TODO: Log loss_q.item() and **loss_info

        if timer % policy_delay == 0:
            for p in self.q_params:
                p.requires_grad = False

            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data['obs'])
            loss_pi.backward()
            self.pi_optimizer.step()

            for p in self.q_params:
                p.requires_grad = True

            # TODO log loss_pi.item()

            while torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, rng):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * rng.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(self):
        for j in range(num_test_episodes):
            episode = env.new_episode()
            o, d, ep_ret, ep_len = episode.get_obs(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                o, r, d, _ = episode.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            # TODO: log ep_ret, rep_len

    def epoch(self):
        for t in range(steps_per_epoch):
            if t < start_steps:
                a = self.env.action_space.sample(rng)
            else:
                a = self.get_action(o, act_noise)

            o2, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            if ep_len == max_ep_len:
                d = False

            self.replay_buffer.store(o, a, r, o2, d)

            o = o2

            if d or (ep_len == max_ep_len):
                # TODO: log ep_ret, ep_len
                episode = env.new_episode()
                o, ep_ret, ep_len = episode.get_obs(), 0, 0

            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = self.replay_buffer.sample_batch(batch_size)
                    self.update(data=batch, timer=j)

        self.test_agent()
