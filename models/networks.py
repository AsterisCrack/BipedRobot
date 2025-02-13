import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal


class PolicyNetwork(nn.Module):
    """
    Policy network
    :param env: (gym Environment) environment actor is operating on
    :param layer1: (int) size of the first hidden layer (default = 100)
    :param layer2: (int) size of the first hidden layer (default = 100)
    """
    def __init__(self, env, layer1=256, layer2=256):
        super(PolicyNetwork, self).__init__()
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        self.action_range = torch.from_numpy(env.action_space.high)
        
        self.lin1 = nn.Linear(self.state_shape, layer1, True)
        self.lin2 = nn.Linear(layer1, layer2, True)
        self.mean_layer = nn.Linear(layer2, self.action_shape, True)
        self.cholesky_layer = nn.Linear(layer2, self.action_shape, True)
        # self.cholesky_layer = nn.Linear(layer2, int((self.action_shape*self.action_shape + self.action_shape)/2), True)
        self.cholesky = torch.zeros(self.action_shape, self.action_shape)

    def forward(self, states):
        """
        forwards input through the network
        :param states: ([State]) a (batch of) state(s) of the environment
        :return: ([float])([float]) mean and cholesky factorization chosen by policy at given state
        """
        x = F.relu(self.lin1(states))
        x = F.relu(self.lin2(x))
        mean = self.action_range * torch.tanh(self.mean_layer(x))
        cholesky_vector = F.softplus(self.cholesky_layer(x))
        # Make cholesky a diagonal matrix where the diagonal is cholesky_vector
        # If a batch
        if cholesky_vector.dim() == 1:
            cholesky = torch.diag(cholesky_vector)
        else:
            cholesky = torch.diag_embed(cholesky_vector)
        """cholesky = []
        if cholesky_vector.dim() == 1 and cholesky_vector.shape[0] > 1:
            cholesky.append(self.to_cholesky_matrix(cholesky_vector))
        else:
            for a in cholesky_vector:
                cholesky.append(self.to_cholesky_matrix(a))
        cholesky = torch.stack(cholesky)"""
        
        return mean, cholesky

    def action(self, state):
        """
        approximates an action by going forward through the network
        :param state: (State) a state of the environment
        :return: (float) an action of the action space
        """
        with torch.no_grad():
            mean, cholesky = self.forward(state)
            # Reshape cholesky to [12, 12]
            # cholesky = cholesky.view(12, 12)
            action_distribution = MultivariateNormal(mean, scale_tril=cholesky)
            action = action_distribution.sample()
        
        return action

    def eval_step(self, state):
        """
        approximates an action based on the mean output of the network
        :param state: (State) a state of  the environment
        :return: (float) an action of the action space
        """
        with torch.no_grad():
            action, _ = self.forward(state)
        return action

    def to_cholesky_matrix(self, cholesky_vector):
        """
        computes cholesky matrix corresponding to a vector
        :param cholesky_vector: ([float]) vector with n items
        :return: ([[float]]) Square Matrix containing the entries of the
                 vector
        """
        k = 0
        cholesky = torch.zeros(self.action_shape, self.action_shape)
        for i in range(self.action_shape):
            for j in range(self.action_shape):
                if i >= j:
                    cholesky[i][j] = cholesky_vector.item() if self.action_shape == 1 else cholesky_vector[k].item()
                    k = k + 1
        return cholesky

class QNetwork(nn.Module):
    """
    QNetwork class for Q-function network
    :param env: (gym Environment) environment newtwork is operating on
    :param layer1: (int) size of the first hidden layer (default = 200)
    :param layer2: (int) size of the first hidden layer (default = 200)
    """
    def __init__(self, env, layer1=256, layer2=256):
        super(QNetwork, self).__init__()
        self.state_shape = env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        #self.lin1 = nn.Linear(self.state_shape + self.action_shape, layer1, True)
        #self.lin2 = nn.Linear(layer1, layer2, True)
        self.lin1 = nn.Linear(self.state_shape, layer1, True)
        self.lin2 = nn.Linear(layer1 + self.action_shape, layer2, True)
        self.lin3 = nn.Linear(layer2, 1, True)

    def forward(self, state, action):
        """
        Forward function forwarding an input through the network
        :param state: (State) a state of the environment
        :param action: (Action) an action of the environments action-space
        :return: (float) Q-value for the given state-action pair
        """
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(torch.cat((x, action), 1)))
        x = self.lin3(x)
        return x