import torch
import torch.nn as nn
from torch.optim import Adam
from util import mlp, combined_shape

def discounted_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter(
        [1],
        [1, float(-discount)],
        x[::-1], axis=0)[::-1]

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, req, val, logp):
        i = self.ptr
        assert i < self.max_size
        self.obs_buf[i] = obs
        self.act_buf[i] = act
        self.rew_buf[i] = req
        self.val_buf[i] = val
        self.logp_buf[i] = logp
        self.ptr = i+1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf,
                        act=self.act_buf,
                        ret=self.ret_buf,
                        adv=self.adv_buf,
                        logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items() }

class BaseActor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    
class GaussianActor(BaseActor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)
    
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)
        
class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # Using Gaussian actor for Box bounded space
        # If using a Discrete action space, use a Categorical actor instead
        self.pi = GaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        
        self.v = Critic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
    
class PPO:
    def __init__(self, env):
        self.env = env
        self.ac = ActorCritic(env.observation_space, env.action_space)
        self.buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        
    def compute_loss_pi(self, obs, act, adv, logp_old):
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Info about loss
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, obs, ret):
        return ((ac.v(obs) - ret)**2).mean()

    def update(self):
        data = self.buf.get()
        pi_l_old, pi_info_old = self.compute_loss_pi(data['obs'], data['act'], data['adv'], data['logp'])
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data['obs'], data['ret'])

        # train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data['obs'], data['act'], data['adv'], data['logp'])
            kl = mpi_avg(pi_info['kl']) # TODO
            if kl > 1.5 * target_kl:
                log.info('Stopping early due to reaching max kl')
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)
            self.pi_optimizer.step()

        # log StopIter=i

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data['obs'], data['ret'])
            loss_v.backward()
            mpi_avg_grads(self.ac.v)
            self.vf_optimizer.step()

        # Log changes from update
        # To log:
        # pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        # LossPi=pi_l_old, LossV=v_l_old, KL=kl, Entropy=ent, ClipFrac=cf
        # DeltaLossPi=(loss_pi.item() - pi_l_old)
        # DeltaLossV=(loss_v.item() - v_l_old)

    def train(self):
        episode = self.env.new_episode()
        o, ep_ret, ep_len = episode.get_obs(), 0, 0

        for epoch in range(epoch):
            for t in range(self.steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                
                next_o, r, d, _ = episode.step(a)
                ep_ret += r
                ep_len += 1

                self.buf.store(o, a, r, v, logp)
                o = next_o

                timeout = (ep_len == max_ep_len)
                terminal = (d or timeout)
                epoch_ended = (t == self.steps_per_epoch - 1)

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        log.warn("trajectory cut off by epoch")
                    if timeout or epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
                    else:
                        v = 0
                    self.buf.finish_path(v)
                    if terminal:
                        # logger.store(EpRet=ep_ret, EpLen=ep_len)
                        pass
                    episode = self.env.new_episode()
                    o, ep_ret, ep_len = episode.get_obs(), 0, 0

            # TODO Save model

            self.update()

            # Log stuff...
                    



    
        
