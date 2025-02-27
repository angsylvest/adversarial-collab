import time
import cv2

from recoder import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
from utils.train import set_seed_everywhere
from utils.environment import get_agent_types

from overcooked_ai_py.env import OverCookedEnv

from model.utils.model import *

from utils.agent import find_index

import hydra
from omegaconf import DictConfig
from overcooked_ai_py.mdp.constants import *

from torch.utils.tensorboard import SummaryWriter
import datetime

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'Workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.discrete_action = cfg.discrete_action_space
        self.save_replay_buffer = cfg.save_replay_buffer
        # self.env = NormalizedEnv(make_env(cfg.env, discrete_action=self.discrete_action))
        self.config_dict = self.create_config(self.cfg)
        # self.config_dict = {}

        self.env = OverCookedEnv(scenario=self.cfg.env, episode_length=self.cfg.episode_length, config=self.config_dict)

        self.env_agent_types = get_agent_types(self.env)
        self.agent_indexes = find_index(self.env_agent_types, 'ally')
        self.adversary_indexes = find_index(self.env_agent_types, 'adversary')

        self.adversarial = cfg.adversarial 
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # OU Noise settings
        self.num_seed_steps = cfg.num_seed_steps
        self.ou_exploration_steps = cfg.ou_exploration_steps
        self.ou_init_scale = cfg.ou_init_scale
        self.ou_final_scale = cfg.ou_final_scale

        if self.discrete_action:
            cfg.agent.params.obs_dim = self.env.observation_space.n
            cfg.agent.params.action_dim = self.env.action_space.n
            cfg.agent.params.action_range = list(range(cfg.agent.params.action_dim))
        else:
            # Don't use!
            cfg.agent.params.obs_dim = self.env.observation_space[0].shape[0]
            cfg.agent.params.action_dim = self.env.action_space[0].shape[0]
            cfg.agent.params.action_range = [-1, 1]

        cfg.agent.params.agent_index = self.agent_indexes
        cfg.agent.params.critic.input_dim = cfg.agent.params.obs_dim + cfg.agent.params.action_dim

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.common_reward = cfg.common_reward
        obs_shape = [len(self.env_agent_types), cfg.agent.params.obs_dim]
        action_shape = [len(self.env_agent_types), cfg.agent.params.action_dim if not self.discrete_action else 1]
        reward_shape = [len(self.env_agent_types), 1]
        dones_shape = [len(self.env_agent_types), 1]
        self.replay_buffer = ReplayBuffer(obs_shape=obs_shape,
                                          action_shape=action_shape,
                                          reward_shape=reward_shape,
                                          dones_shape=dones_shape,
                                          capacity=int(cfg.replay_buffer_capacity),
                                          device=self.device)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

        self.players = self.env.overcooked.state.players
        # self.bayes_buffer = {"alpha_lazy": 1, "alpha_adv": 1, "beta_lazy": 1, "beta_adv": 1}

        # create relevant tensorboard params to record 
        self.sw = SummaryWriter(log_dir=os.path.join(self.work_dir, f"../../../runs/{self.cfg.env}{save_path_include}/{date_str}_experiment_{self.cfg.env}_alg_{alg_type}_lazy_{lazy_agent, lazy_prob}_adver_{adv_agent, advers_prob}_include_{include_in}"))# SummaryWriter(log_dir=os.path.join(self.work_dir, 'runs'))

    def create_config(self, cfg):
        if not cfg: 
            return {}

        self.params = {
            "alg_type": cfg.alg_type, # "multiTravos" # [multiTravos, baseline, Travos]
            "lazy_agent": cfg.lazy_agent,
            "adv_agent": cfg.adv_agent, 
            "both": cfg.both, 
            "advers_prob": cfg.advers_prob,
            "lazy_prob": cfg.lazy_prob,
            "discretize_trust": cfg.discretize_trust,  
            "adaptive_discretize": cfg.adaptive_discretize, 
            "include_in": cfg.include_in, 
            "one_hot_encode": cfg.one_hot_encode,  
            "include_thres": cfg.include_thres, 
            "include_trust": True,  
            "multi_dim_trust": True,  
            "save_path_include": "" 
        } 

        if cfg.lazy_agent and cfg.adv_agent:
            self.params["both"] = True 

        alg_type = cfg.alg_type

        if cfg.alg_type == "baseline": 
            self.params["include_trust"] = False # True # Only TRAVOS
            self.params["multi_dim_trust"] = False # True # For multi-TRAVOS
        elif cfg.alg_type == "Travos": 
            include_thres = False 
            alg_type += f"_disc_{discretize_trust}_adapt_{adaptive_discretize}"
            self.params["include_trust"] = True # True # Only TRAVOS
            self.params["multi_dim_trust"] = False # True # For multi-TRAVOS    
        else: 
            alg_type += f"_disc_{discretize_trust}_adapt_{adaptive_discretize}_int_count_{cfg.include_thres}"
            self.params["include_trust"] = True # True # Only TRAVOS
            self.params["multi_dim_trust"] = True # True # For multi-TRAVOS    

        if lazy_agent: 
            self.params["save_path_include"] = f"_lazy_{lazy_prob}"

        if adv_agent: 
            self.params["save_path_include"] = f"_adv_{advers_prob}"

        self.params["alg_type"] = alg_type 
        return self.params

    def evaluate(self):
        average_episode_reward = 0

        self.video_recorder.init(enabled=True)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            episode_step = 0

            done = False
            episode_reward = 0
            while not done:
                if self.adversarial:
                    agent_observation = obs[0].reshape(1, -1) # [self.agent_indexes]
                    agent_actions = self.agent.act(agent_observation, sample=False)

                    action = agent_actions
                    other_agent_action = torch.tensor(np.array([self.env.action_space.sample()]), dtype=torch.float32).reshape(1, 1)

                    if not isinstance(action, torch.Tensor):
                        action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)

                    action = torch.cat((action, other_agent_action), dim = 0)
                else: 
                    # agent_observation = obs[self.agent_indexes]
                    # agent_actions = self.agent.act(agent_observation, sample=False)
                    # action = agent_actions
                    action = self.agent.act(obs, sample=False)

                obs, rewards, done, info = self.env.step(action)

                rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)

                self.video_recorder.record(self.env)

                episode_reward += sum(rewards)[0]
                episode_step += 1

            average_episode_reward += episode_reward
        self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.sw.add_scalar("Reward/eval", average_episode_reward, self.step)
        self.sw.flush()
        self.logger.dump(self.step)

    def run(self):
        info = None 
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps + 1:
            if done or self.step % self.cfg.eval_frequency == 0:
                # print(f'interactions dict: {self.env.overcooked.state.players[0].interaction_history}')

                if self.step > 0:
                    # example of dump step 
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))
                    self.sw.flush() # will ensure paramters are written to disk 

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                    start_time = time.time()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.sw.add_scalar("Reward/train", episode_reward, self.step)

                # angel: add custom logger info here 
                if include_trust: 
                    self.logger.log('perf/episode', episode, self.step)
                    self.logger.log('perf/alpha_laz', self.env.overcooked.state.players[0].alpha_lazy, self.step)
                    self.logger.log('perf/beta_laz', self.env.overcooked.state.players[0].beta_lazy, self.step)
                    self.logger.log('perf/trust_laz', self.env.overcooked.state.players[0].interaction_history["lazy_trust"], self.step)
                    self.logger.log('perf/uncert_laz', self.env.overcooked.state.players[0].interaction_history["lazy_uncert"], self.step)

                    self.sw.add_scalar("Trust/lazy", self.env.overcooked.state.players[0].interaction_history["lazy_trust"], self.step)
                    self.sw.add_scalar("Trust/lazy_uncert", self.env.overcooked.state.players[0].interaction_history["lazy_uncert"], self.step)
                    
                    if multi_dim_trust: 
                        self.logger.log('perf/episode', episode, self.step )
                        self.logger.log('perf/alpha_adv', self.env.overcooked.state.players[0].alpha_adversary, self.step)
                        self.logger.log('perf/beta_adv', self.env.overcooked.state.players[0].beta_adversary, self.step)
                        self.logger.log('perf/trust_adv', self.env.overcooked.state.players[0].interaction_history["adv_trust"], self.step)
                        self.logger.log('perf/uncert_adv', self.env.overcooked.state.players[0].interaction_history["adv_uncert"], self.step)
                
                        self.sw.add_scalar("Trust/adv", self.env.overcooked.state.players[0].interaction_history["adv_trust"], self.step)
                        self.sw.add_scalar("Trust/adv_uncert", self.env.overcooked.state.players[0].interaction_history["adv_uncert"], self.step)
                    
                    # print(f'curr interaction history: {self.env.overcooked.state.players[0].interaction_history}')
                    # self.sw.flush()

                self.sw.flush() # will ensure paramters are written to disk 

                obs = self.env.reset()

                self.ou_percentage = max(0, self.ou_exploration_steps - (self.step - self.num_seed_steps)) / self.ou_exploration_steps
                self.agent.scale_noise(self.ou_final_scale + (self.ou_init_scale - self.ou_final_scale) * self.ou_percentage)
                self.agent.reset_noise()

                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

                # self.logger.dump(self.step)

            if self.step < self.cfg.num_seed_steps:
                action = np.array([self.env.action_space.sample() for _ in self.env_agent_types])
                if self.discrete_action: action = action.reshape(-1, 1)
            else:
                if self.adversarial:
                    agent_observation = obs[0].reshape(1, -1) # [self.agent_indexes]
                    agent_actions = self.agent.act(agent_observation, sample=True)

                    action = agent_actions
                    other_agent_action = torch.tensor(np.array([self.env.action_space.sample()]), dtype=torch.float32).reshape(1, 1)

                    if not isinstance(action, torch.Tensor):
                        action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)

                    action = torch.cat((action, other_agent_action), dim = 0)
                else: 
                    agent_observation = obs[self.agent_indexes]
                    agent_actions = self.agent.act(agent_observation, sample=True)
                    action = agent_actions

            if self.step >= self.cfg.num_seed_steps and self.step >= self.agent.batch_size:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, rewards, done, info = self.env.step(action, info)
            # print(f'next obs: {next_obs}
            
            rewards = np.array(info['shaped_r_by_agent']).reshape(-1, 1)


            if episode_step + 1 == self.env.episode_length:
                done = True

            if self.cfg.render:
                cv2.imshow('Overcooked', self.env.render())
                cv2.waitKey(1)

            episode_reward += sum(rewards)[0]

            if self.discrete_action: action = action.reshape(-1, 1)

            dones = np.array([done for _ in self.env.agents]).reshape(-1, 1)

            self.replay_buffer.add(obs, action, rewards, next_obs, dones)

            obs = next_obs
            episode_step += 1
            self.step += 1

            if self.step % 5e4 == 0 and self.save_replay_buffer:
                self.replay_buffer.save(self.work_dir, self.step - 1)


@hydra.main(config_path='config', config_name='train')
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
