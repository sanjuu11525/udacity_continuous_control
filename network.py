import torch
import torch.nn as nn


class DeterministicCriticNet(nn.Module):
    """Deterministic Critic Model."""

    def __init__(self, state_size, action_size, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DeterministicCriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(inplace=True),
                                 nn.Linear(128       , 128), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(128 + action_size, 64), nn.ReLU(inplace=True),
                                 nn.Linear(64, 1))
    def forward(self, state, action):
        x = self.fc1(state)
        x = self.fc2(torch.cat([x, action], dim=1))
        return x


class DeterministicActorNet(nn.Module):
    """Deterministic Actor Model."""

    def __init__(self, state_size, action_size, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DeterministicActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.model = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU(inplace=True),
                                   nn.Linear(128       , 256), nn.ReLU(inplace=True),
                                   nn.Linear(256       , action_size), nn.Tanh())

    def forward(self, state):
        return self.model(state)