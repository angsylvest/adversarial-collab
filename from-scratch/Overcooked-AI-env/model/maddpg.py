import torch
import numpy as np

from utils.misc import soft_update

from model.DDPGAgent import DDPGAgent
from model.utils.model import *
from overcooked_ai_py.mdp.constants import *

class MADDPG(object):

    def __init__(self, name, params):

        self.name = name
        self.lr = params.lr
        self.gamma = params.gamma
        self.tau = params.tau

        self.obs_dim = params.obs_dim
        self.action_dim = params.action_dim
        self.batch_size = params.batch_size
        self.device = params.device
        self.discrete_action = params.discrete_action_space

        self.agent_index = params.agent_index
        self.num_agents = len(self.agent_index)

        self.mse_loss = torch.nn.MSELoss()

        # Reshape critic input shape for shared observation
        params.critic.obs_dim = (self.obs_dim + self.action_dim) * self.num_agents

        self.agents = [DDPGAgent(params) for _ in range(self.num_agents)]
        [agent.to(self.device) for agent in self.agents]

    def scale_noise(self, scale):
        for agent in self.agents:
            agent.scale_noise(scale)

    def reset_noise(self):
        for agent in self.agents:
            agent.reset_noise()
            agent.reset_hidden()

    def act(self, observations, sample=False):
        observations = torch.Tensor(observations).to(self.device)

        actions = []
        for agent, obs in zip(self.agents, observations):
            agent.eval()
            actions.append(agent.act(obs, explore=sample).squeeze())
            agent.train()
        return np.array(actions)

    def update(self, replay_buffer, logger, step):

        sample = replay_buffer.sample(self.batch_size, nth=self.agent_index)
        obses, actions, rewards, next_obses, dones = sample

        if self.discrete_action:  actions = number_to_onehot(actions)

        for agent_i, agent in enumerate(self.agents):

            ''' Update value '''
            agent.critic_optimizer.zero_grad()

            with torch.no_grad():
                if self.discrete_action:
                    target_actions = torch.Tensor([onehot_from_logits(policy(next_obs)).detach().cpu().numpy() for policy, next_obs in
                                               zip(self.target_policies, torch.swapaxes(next_obses, 0, 1))]).to(self.device)
                else:
                    target_actions = torch.Tensor([policy(next_obs).detach().cpu().numpy() for policy, next_obs in
                                                   zip(self.target_policies, torch.swapaxes(next_obses, 0, 1))]).to(self.device)
                target_actions = torch.swapaxes(target_actions, 0, 1)
                target_critic_in = torch.cat((next_obses, target_actions), dim=2).view(self.batch_size, -1)
                target_next_q = rewards[:, agent_i] + (1 - dones[:, agent_i]) * self.gamma * agent.target_critic(target_critic_in)

            critic_in = torch.cat((obses, actions), dim=2).view(self.batch_size, -1)
            main_q = agent.critic(critic_in)

            critic_loss = self.mse_loss(main_q, target_next_q)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

            ''' Update policy '''
            agent.policy_optimizer.zero_grad()

            policy_out = agent.policy(obses[:, agent_i])
            if self.discrete_action:
                action = gumbel_softmax(policy_out, hard=True)
            else:
                action = policy_out

            joint_actions = torch.zeros((self.batch_size, self.num_agents, self.action_dim)).to(self.device)
            for i, policy, local_obs, act in zip(range(self.num_agents), self.policies, torch.swapaxes(obses, 0, 1), torch.swapaxes(actions, 0, 1)):
                if i == agent_i:
                    joint_actions[:, i] = action
                else:
                    other_action = onehot_from_logits(policy(local_obs)) if self.discrete_action else policy(local_obs)
                    joint_actions[:, i] = other_action

            critic_in = torch.cat((obses, joint_actions), dim=2).view(self.batch_size, -1)

            actor_loss = -agent.critic(critic_in).mean()
            actor_loss += (policy_out ** 2).mean() * 1e-3  # Action regularize

            if use_lstm: 
                encoded_obs_seq, _ = agent.observation_encoder(obses[:, agent_i, :, :])  # pass sequence of observations
                temporal_diff = encoded_obs_seq[:, 1:] - encoded_obs_seq[:, :-1]
                temporal_loss = torch.mean(torch.norm(temporal_diff, dim=-1))
                alpha = 0.1
                actor_loss += alpha * temporal_loss

                # kl-divergence loss 
                # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                # beta = 0.01  # KL weight coefficient, tune as needed
                # actor_loss += beta * kl_loss
            
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
            agent.policy_optimizer.step()

        self.update_all_targets()

    def update_all_targets(self):
        for agent in self.agents:
            soft_update(agent.target_critic, agent.critic, self.tau)
            soft_update(agent.target_policy, agent.policy, self.tau)

    def save(self, step):
        # os.mk
        #
        # for i, agent in self.agents:
        #     name = '{0}_{1}_{step}.pth'.format(self.name, i, step)
        #     torch.save(agent, )
        #
        #
        # raise NotImplementedError
        pass

    def load(self, filename):

        raise NotImplementedError

    @property
    def policies(self):
        return [agent.policy for agent in self.agents]

    @property
    def target_policies(self):
        return [agent.target_policy for agent in self.agents]

    @property
    def critics(self):
        return [agent.critic for agent in self.agents]

    @property
    def target_critics(self):
        return [agent.target_critic for agent in self.agents]
