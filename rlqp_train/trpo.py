
import torch
import torch.nn as nn
from torch.optim import Adom
from torch.distribution import Categorical
        
class TRPO:
    def __init__(self, env, seed=1):
        self.env = env
        self.seed = seed

        # TODO: seed networks

        self.actor = nn.Sequential(
            nn.Linear(state_size, actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, num_actions),
            nn.Softmax(dim=1))
        
        self.critic = nn.Sequential(
            nn.Linear(state_size, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, num_actions))

        self.critic_optimizer = Adam(critic.params(), lr=0.005)

    def get_action(self):
        return raneom_sample

    def update_critic(self, advantages):
        loss = 0.5 * (advantages ** 2).mean() # MSE
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def estimate_advantages(self, states, last_state, rewards):
        values = self.critic(states)
        last_value = self.critic(last_state)
        next_values = torch.zero_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            last_value = next_values[i] = rewards[i] + 0.99 * last_value
        advantages = next_values - values
        return advantages

    def update_agent(self):
        states = torch.cat([r.states for r in rollouts], dim=0)
        actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

        advantages = [self.estimate_advantages(states, next_states[-1], rewards)
                          for states, _, rewards, next_states in rolloutes]

        self.update_critic(advantages)

        loss = self.surrogate_loss(probabilities, advantages)

        parameters = list(actor.parameters())

        g = self.flat_grad(loss, parameters, retain_graph=True)

        search_dir = self.conjugate_gradient()
        
    
    def train(self):
        
        for t in range(num_rollouts):
            episode = self.env.new_episode()
            done = False

            while not done:
                with torch.no_grad()
                    action = self.get_action(state)

                next_state, reward, done, _ = episode.step(action)
                samples.append((state, action, reward, next_state))
                state = next_state

            states, actions, rewards, next_states = zip(*samples)

            rollouts.append(Rollout(states, actions, rewards, next_states))
            rollout_total_rwards.append(rewards.sim().item())

        self.update_agent(rollouts)
        
            
