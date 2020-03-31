import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from rl_da.da_network import DANetwork, DAValueNetwork


class ReinforceEstimator:
    def __init__(self, observation_dim, n_observations, max_landmarks, hidden_state_size, gamma=0.5, lr=1e-3):
        self.observation_dim_ = observation_dim
        self.n_observations_ = n_observations
        self.max_landmarks_ = max_landmarks
        self.hidden_state_size_ = hidden_state_size
        self.gamma_ = gamma
        self.lr_ = lr

        self.da_net_ = DANetwork(n_observations * observation_dim, 10, n_observations + 1, max_landmarks)
        self.optimizer_ = optim.Adam(self.da_net_.parameters(), lr=lr)

    def get_action(self, state):
        association_probabilities = self.da_net_.forward(torch.Tensor(state))

        actions = []
        log_probability = None

        for probabilities in association_probabilities:
            observation_index = probabilities.argmax().detach().numpy()
            actions.append(observation_index)
            log_probability = torch.log(probabilities[observation_index]) if log_probability is None else \
                log_probability + torch.log(probabilities[observation_index])

        return np.array(actions), log_probability

    def update_policy(self, rewards, log_probabilities):
        returns = self.get_returns_(rewards)
        policy_gradient = []
        # TODO: Check
        for log_probability, Gt in zip(log_probabilities, returns):
            policy_gradient.append(-log_probability * Gt)
        loss = torch.stack(policy_gradient).sum()
        self.optimizer_.zero_grad()
        loss.backward()
        self.optimizer_.step()

    def get_returns_(self, rewards):
        returns = []
        Gt = 0
        pw = 0
        for reward in rewards[::-1]:
            Gt += self.gamma_ ** pw * reward
            pw += 1
            returns.append(Gt)
        returns = returns[::-1]
        returns = torch.tensor(returns, requires_grad=True)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns


class A2CEstimator:
    def __init__(self, observation_dim, n_observations, max_landmarks, hidden_state_size, gamma=0.5, lr=1e-3):
        self.observation_dim_ = observation_dim
        self.n_observations_ = n_observations
        self.max_landmarks_ = max_landmarks
        self.hidden_state_size_ = hidden_state_size
        self.gamma_ = gamma
        self.lr_ = lr

        self.da_net_ = DAValueNetwork(n_observations * observation_dim, 10, n_observations + 1, max_landmarks,
                                      max_landmarks * n_observations, 20)

        self.optimizer_ = optim.Adam(self.da_net_.parameters(), lr=lr)

    def get_action(self, state):
        state = torch.Tensor(state)
        association_probabilities, state_value = self.da_net_.forward(state)

        actions = []
        log_probability = None

        for probabilities in association_probabilities:
            observation_index = probabilities.argmax().detach().numpy()
            actions.append(observation_index)
            log_probability = torch.log(probabilities[observation_index]) if log_probability is None else \
                log_probability + torch.log(probabilities[observation_index])

        return np.array(actions), log_probability, state_value

    def update_policy(self, rewards, log_probabilities, state_values):
        returns = self.get_returns_(rewards)
        policy_grad = []
        value_grad = []
        for log_probability, value, Gt in zip(log_probabilities, state_values, returns):
            advantage = Gt - value
            policy_grad.append(-log_probability * advantage)
            value_grad.append(F.smooth_l1_loss(value, Gt))
        policy_loss = torch.stack(policy_grad).sum()
        value_loss = torch.stack(value_grad).sum()
        loss = policy_loss + value_loss

        self.optimizer_.zero_grad()
        loss.backward()
        self.optimizer_.step()

    def get_returns_(self, rewards):
        returns = []
        Gt = 0
        pw = 0
        for reward in rewards[::-1]:
            Gt += self.gamma_ ** pw * reward
            pw += 1
            returns.append(Gt)
        returns = returns[::-1]
        returns = torch.tensor(returns, requires_grad=True)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

