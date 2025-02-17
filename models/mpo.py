import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from scipy.optimize import minimize

from models.networks import PolicyNetwork, QNetwork
from tensorboardX import SummaryWriter


class MPO(object):
    """
    Maximum cholesky Posteriori Policy Optimization (MPO)

    :param env: (Gym Environment) gym environment to learn on
    :param dual_constraint: (float) hard constraint of the dual formulation in the E-step
    :param mean_constraint: (float) hard constraint of the mean in the M-step
    :param var_constraint: (float) hard constraint of the covariance in the M-step
    :param learning_rate: (float) learning rate in the Q-function
    :param alpha: (float) scaling factor of the lagrangian multiplier in the M-step
    :param episodes: (int) number of training (evaluation) episodes
    :param episode_length: (int) step size of one episode
    :param lagrange_it: (int) number of optimization steps of the Lagrangian
    :param mb_size: (int) size of the sampled mini-batch
    :param sample_episodes: (int) number of sampling episodes
    :param add_act: (int) number of additional actions
    :param policy_layers: (tuple) size of the hidden layers in the policy_net net
    :param critic_layers: (tuple) size of the hidden layers in the critic net
    :param log: (boolean) saves log if True
    :param log_dir: (str) directory in which log is saved
    :param render: (boolean) renders the simulation if True
    :param save: (boolean) saves the model if True
    :param save_path: (str) path for saving and loading a model
    """
    def __init__(self,
                 env,
                 policy_net: PolicyNetwork,
                 target_policy_net: PolicyNetwork,
                 q_net: QNetwork,
                 target_q_net: QNetwork,
                 policy_optimizer=None, 
                 q_optimizer=None,
                 dual_constraint=0.1, mean_constraint=0.1, var_constraint=1e-4,
                 learning_rate=0.99, alpha=10, episodes=int(200), episode_length=3000,
                 lagrange_it=5, mb_size=64, rerun_mb=5, sample_episodes=1, add_act=64,
                 log=True, log_dir=None, render=False, save=True, save_path="mpo_model.pt"):
        # initialize env
        self.env = env

        # initialize some hyperparameters
        self.alpha = alpha  # scaling factor for the update step of lagrange_mean
        self.epsilon = dual_constraint  # hard constraint for the KL
        self.epsilon_mean = mean_constraint  # hard constraint for the KL
        self.epsilon_covariance = var_constraint  # hard constraint for the KL
        self.lr = learning_rate  # learning rate
        self.episodes = episodes
        self.episode_length = episode_length
        self.lagrange_it = lagrange_it
        self.mb_size = mb_size
        self.rerun_mb = rerun_mb
        self.M = add_act
        self.action_shape = env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)

        # initialize networks
        self.q_net = q_net
        self.target_q_net = target_q_net
        
        self.policy_net = policy_net
        self.target_policy_net = target_policy_net
        
        # Copy target parameters to trained parameter
        for target_param, param in zip(self.target_q_net.parameters(),
                                       self.q_net.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        
        for target_param, param in zip(self.target_policy_net.parameters(),
                                       self.policy_net.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
                
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=3e-4) if q_optimizer is None else q_optimizer
        self.policy_net_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4) if policy_optimizer is None else policy_optimizer
        self.mse_loss = nn.MSELoss()

        # initialize Lagrange Multiplier
        self.lagrange = np.random.rand()
        self.lagrange_mean = np.random.rand()
        self.lagrange_covariance = np.random.rand()

        # control/log variables
        self.episode = 0
        self.sample_episodes = sample_episodes
        self.log = log
        self.log_dir = log_dir
        self.render = render
        self.save = save
        self.save_path = save_path

    def _sample_trajectory(self, episodes, episode_length, render):
        """
        Samples a trajectory which serves as a batch
        :param episodes: (int) number of episodes to be sampled
        :param episode_length: (int) length of a single episode
        :param render: (bool) flag if steps should be rendered
        :return: [States], [Action], [Reward], [State]: batch of states, actions, rewards and next-states
        """
        states = []
        rewards = []
        actions = []
        next_states = []
        mean_reward = 0
        for episode in range(episodes):
            observation, info = self.env.reset()
            for step in range(episode_length):
                action = np.reshape(
                    self.target_policy_net.action(torch.from_numpy(observation).float()).numpy(),
                    -1)
                new_observation, reward, done, _, info = self.env.step(action)
                mean_reward += reward
                if render:
                    self.env.render()
                states.append(observation)
                rewards.append(reward)
                actions.append(action)
                next_states.append(new_observation)
                if done:
                    observation, info = self.env.reset()
                else:
                    observation = new_observation
        mean_reward = mean_reward / episode_length / episodes
        states = np.array(states)
        # states = torch.tensor(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        return states, actions, rewards, next_states, mean_reward

    def _q_net_update(self, states, rewards, actions, mean_next_q):
        """
        Updates the q networks
        :param states: ([State]) mini-batch of states
        :param actions: ([Action]) mini-batch of actions
        :param rewards: ([Reward]) mini-batch of rewards
        :param mean_next_q: ([State]) target Q values
        :return: (float) q-loss
        """
        rewards = torch.from_numpy(rewards).float()
        y = rewards + self.lr * mean_next_q
        self.q_net_optimizer.zero_grad()
        target = self.q_net(torch.from_numpy(states).float(), torch.from_numpy(actions).float())
        loss_critic = self.mse_loss(y, target)
        loss_critic.backward()
        self.q_net_optimizer.step()
        return loss_critic.item()

    def _calculate_gaussian_kl(self, policy_mean, target_mean, policy_cholesky, target_cholesky):
        """
        calculates the KL between the old and new policy assuming a gaussian distribution
        :param policy_mean: ([float]) mean of the policy_net
        :param target_mean: ([float]) mean of the target policy_net
        :param policy_cholesky: ([[float]]) cholesky matrix of the policy_net covariance
        :param target_cholesky: ([[float]]) cholesky matrix of the target policy_net covariance
        :return: C_mean, C_covariance: ([float],[[float]])mean and covariance terms of the KL
        """
        inner_covariance = []
        inner_mean = []
        
        for mean, target_mean, a, target_a in zip(policy_mean, target_mean, policy_cholesky, target_cholesky):
            covariance = a @ a.t()
            target_covariance = target_a @ target_a.t()
            inverse = covariance.inverse()
            inner_covariance.append(torch.trace(inverse @ target_covariance)
                           - covariance.size(0)
                           + torch.log(covariance.det() / target_covariance.det()))
            inner_mean.append((mean - target_mean) @ inverse @ (mean - target_mean))

        inner_mean = torch.stack(inner_mean)
        inner_covariance = torch.stack(inner_covariance)
        
        C_mean = 0.5 * torch.mean(inner_covariance)
        C_covariance = 0.5 * torch.mean(inner_mean)
        return C_mean, C_covariance

    def _update_param(self):
        """
        Sets target parameters to trained parameter
        """
        # Update policy parameters
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        # Update critic parameters
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)

    def train(self, episodes=None, episode_length=None, sample_episodes=None, rerun_mb=None,
              render=None, save=None, save_path=None, log=None, log_dir=None):
        """
        Trains a model based on MPO
        :param episodes: (int) number of training (evaluation) episodes
        :param episode_length: (int) step size of one episode
        :param sample_episodes: (int) number of sampling episodes
        :param rerun_mb: (int) number of times the episode is used for evaluation
        :param render: (boolean) renders the simulation if True
        :param save: (boolean) saves the model if True
        :param save_path: (str) path for saving and loading a model
        :param log: (boolean) saves log if True
        :param log_dir: (str) directory in which log is saved
        """
        # initialize flags and params
        render = render if render is not None else self.render
        save = save if save is not None else self.save
        save_path = save_path if save_path is not None else self.save_path
        episodes = episodes if episodes is not None else self.episodes
        episode_length = episode_length if episode_length is not None else self.episode_length
        sample_episodes = sample_episodes if sample_episodes is not None else self.sample_episodes
        rerun_mb = rerun_mb if rerun_mb is not None else self.rerun_mb

        # initialize logging
        log = log if log is not None else self.log
        log_dir = log_dir if log_dir is not None else self.log_dir
        if log:
            writer = SummaryWriter() if log_dir is None else SummaryWriter("runs/" + log_dir)

        # start training
        for episode in range(self.episode, episodes):

            # Update replay buffer
            states, actions, rewards, next_states, mean_reward = self._sample_trajectory(sample_episodes, episode_length, render)
            mean_q_loss = 0
            mean_lagrange = 0

            # Find better policy by gradient descent
            for _ in range(rerun_mb):
                for indices in BatchSampler(SubsetRandomSampler(range(episode_length)), self.mb_size, False):
                    state_batch = states[indices]
                    action_batch = actions[indices]
                    reward_batch = rewards[indices]
                    next_state_batch = next_states[indices]

                    # sample M additional action for each state
                    target_mean, target_cholesky = self.target_policy_net.forward(torch.tensor(state_batch).float())
                    target_mean.detach()
                    target_cholesky.detach()
                    action_distribution = MultivariateNormal(target_mean, scale_tril=target_cholesky)
                    additional_action = []
                    additional_target_q = []
                    additional_next_q = []
                    additional_q = []
                    for _ in range(self.M):
                        action = action_distribution.sample()
                        additional_action.append(action)
                        additional_target_q.append(self.target_q_net.forward(torch.tensor(state_batch).float(),
                                                                              action).detach().numpy())
                        additional_next_q.append(self.target_q_net.forward(torch.tensor(next_state_batch).float(),
                                                                            action).detach())
                        additional_q.append(self.q_net.forward(torch.tensor(state_batch).float(),
                                                                action))
                    # print(additional_action)
                    additional_action = torch.stack(additional_action).squeeze()
                    additional_q = torch.stack(additional_q).squeeze()
                    additional_target_q = np.array(additional_target_q).squeeze()
                    additional_next_q = torch.stack(additional_next_q).squeeze()

                    mean_q = torch.mean(additional_q, 0)
                    mean_next_q = torch.mean(additional_next_q, 0)

                    # Update Q-function
                    q_loss = self._q_net_update(
                        states=state_batch,
                        rewards=reward_batch,
                        actions=action_batch,
                        mean_next_q=mean_next_q
                    )
                    mean_q_loss += q_loss   # TODO: can be removed

                    # E-step
                    # Update Dual-function
                    def dual(lagrange):
                        """
                        Dual function of the non-parametric variational
                        g(lagrange) = lagrange*epsilon + lagrange \sum \log (\sum \exp(Q(a, s)/lagrange))
                        """
                        max_q = np.max(additional_target_q, 0)
                        return lagrange * self.epsilon + np.mean(max_q) \
                            + lagrange * np.mean(np.log(np.mean(np.exp((additional_target_q - max_q) / lagrange), 0)))

                    bounds = [(1e-6, None)]
                    res = minimize(dual, np.array([self.lagrange]), method='SLSQP', bounds=bounds)
                    self.lagrange = res.x[0]

                    # calculate the new q values
                    exp_Q = torch.tensor(additional_target_q) / self.lagrange
                    baseline = torch.max(exp_Q, 0)[0]
                    exp_Q = torch.exp(exp_Q - baseline)
                    normalization = torch.mean(exp_Q, 0)
                    
                    # Reshape exp_Q and normalization to be broadcastable with additional_action
                    exp_Q = exp_Q.unsqueeze(-1)  # Shape: [10, 64, 1]
                    normalization = normalization.unsqueeze(-1)  # Shape: [64, 1]

                    action_q = additional_action * exp_Q / normalization
                    action_q = np.clip(action_q, a_min=-self.action_range,
                                a_max=self.action_range)

                    # M-step
                    # update policy based on lagrangian
                    for _ in range(self.lagrange_it):
                        mean, cholesky = self.policy_net.forward(torch.tensor(state_batch).float())
                        policy = MultivariateNormal(mean, scale_tril=cholesky)

                        additional_logprob = []
                        if self.M == 1:
                            additional_logprob = policy.log_prob(action_q)
                        else:
                            for column in range(self.M):
                                action_vec = action_q[column, :]
                                additional_logprob.append(policy.log_prob(action_vec))
                            additional_logprob = torch.stack(additional_logprob).squeeze()

                        C_mean, C_covariance = self._calculate_gaussian_kl(policy_mean=mean,
                                                               target_mean=target_mean,
                                                               policy_cholesky=cholesky,
                                                               target_cholesky=target_cholesky)

                        # Update lagrange multipliers by gradient descent
                        self.lagrange_mean -= self.alpha * (self.epsilon_mean - C_mean).detach().item()
                        self.lagrange_covariance -= self.alpha * (self.epsilon_covariance - C_covariance).detach().item()

                        if self.lagrange_mean < 0:
                            self.lagrange_mean = 0
                        if self.lagrange_covariance < 0:
                            self.lagrange_covariance = 0

                        self.policy_net_optimizer.zero_grad()
                        loss_policy = -(
                                torch.mean(additional_logprob)
                                + self.lagrange_mean * (self.epsilon_mean - C_mean)
                                + self.lagrange_covariance * (self.epsilon_covariance - C_covariance)
                        )
                        mean_lagrange += loss_policy.item()
                        loss_policy.backward()
                        self.policy_net_optimizer.step()

            self._update_param()
            
            mean_q_loss = mean_q_loss / self.rerun_mb
            mean_lagrange = mean_lagrange / self.lagrange_it / self.rerun_mb
            
            print(
                "\n Episode:\t", episode,
                "\n Mean reward:\t", mean_reward,
                "\n Mean Q loss:\t", mean_q_loss,
                "\n Mean Lagrange:\t", mean_lagrange,
                "\n lagrange:\t", self.lagrange,
                "\n lagrange_mean:\t", self.lagrange_mean,
                "\n lagrange_covariance:\t", self.lagrange_covariance,
            )

            # saving and logging
            if save is True:
                self.save_model(episode=episode, path=save_path)
            if log:
                number_mb = int(self.episode_length / self.mb_size) + 1
                reward_target = self.eval(10, episode_length, render=False)
                writer.add_scalar('target/mean_rew_10_ep', reward_target,
                                  episode + 1)
                writer.add_scalar('data/mean_reward', mean_reward, episode + 1)
                writer.add_scalar('data/mean_lagrangeloss', mean_lagrange / number_mb, episode + 1)
                writer.add_scalar('data/mean_qloss', mean_q_loss / number_mb, episode + 1)

        # end training
        if log:
            writer.close()

    def eval(self, episodes, episode_length, render=True):
        """
        Method for evaluating current model (mean reward for a given number of episodes and episode length)
        :param episodes: (int) number of episodes for the evaluation
        :param episode_length: (int) length of a single episode
        :param render: (bool) flag if to render while evaluating
        :return: (float) meaned reward achieved in the episodes
        """
        total_rewards = 0
        for episode in range(episodes):
            reward = 0
            observation, info = self.env.reset()
            for step in range(episode_length):
                action = self.target_policy_net.eval_step(torch.tensor(observation).float()).numpy()
                new_observation, rew, done, _, info = self.env.step(action)
                reward += rew
                if render:
                    self.env.render()
                observation = new_observation if not done else self.env.reset()[0]

            total_rewards += reward
        return total_rewards/episodes

    def load_model(self, path=None):
        """
        Loads a model from a given path
        :param path: (str) file path (.pt file)
        """
        load_path = path if path is not None else self.save_path
        checkpoint = torch.load(load_path)
        self.episode = checkpoint['epoch']
        self.q_net.load_state_dict(checkpoint['critic_state_dict'])
        self.target_q_net.load_state_dict(checkpoint['target_critic_state_dict'])
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_policy_net.load_state_dict(checkpoint['target_policy_state_dict'])
        self.q_net_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.policy_net_optimizer.load_state_dict(checkpoint['policy_optim_state_dict'])
        self.q_net.train()
        self.target_q_net.train()
        self.policy_net.train()
        self.target_policy_net.train()

    def save_model(self, episode=0, path=None):
        """
        Saves the model
        :param episode: (int) number of learned episodes
        :param path: (str) file path (.pt file)
        """
        safe_path = path if path is not None else self.save_path
        data = {
            'epoch': episode,
            'critic_state_dict': self.q_net.state_dict(),
            'target_critic_state_dict': self.target_q_net.state_dict(),
            'policy_state_dict': self.policy_net.state_dict(),
            'target_policy_state_dict': self.target_policy_net.state_dict(),
            'critic_optim_state_dict': self.q_net_optimizer.state_dict(),
            'policy_optim_state_dict': self.policy_net_optimizer.state_dict()
        }
        torch.save(data, safe_path)