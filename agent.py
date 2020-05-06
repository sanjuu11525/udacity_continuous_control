import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple, deque

from network import *

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device, seed=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class Sampler:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, max_sigma, min_sigma, end_episode):
        """Initialize parameters for noise generation."""
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.end_episode = end_episode

    def sample(self, i_episode, action_values):
        """Generate noise based on action values with decayed sigma."""
        sigma = max(self.max_sigma + (self.min_sigma - self.max_sigma)/self.end_episode * i_episode, self.min_sigma) 
        return np.random.normal(action_values, sigma)


class DeterministicACagent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, params, device, seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.params = params
        self.seed = seed
        self.device = device
        
        self.noise = Sampler(params.max_sigma, params.min_sigma, params.end_episode)
        
        # Q-Network
        self.critic_local = DeterministicCriticNet(state_size, action_size)
        self.critic_target = DeterministicCriticNet(state_size, action_size)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.params.lr)
        
        self.actor_local = DeterministicActorNet(state_size, action_size)
        self.actor_target = DeterministicActorNet(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.params.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size=action_size     , buffer_size=params.buffer_size,
                                       batch_size=params.batch_size, device=device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def load_actor(self, checkpoint):
        """Load actor for inference only.
        
        Params
        ======
            checkpoint (string): model path
        """
        
        model = torch.load(checkpoint)
        self.actor_local.load_state_dict(model)
                        
    def step(self, state, action, reward, next_state, done):
        """Queue experience in reply memory and make train the model.
        
        Params
        ======
            state (float): current state
            action (int): action to next state
            reward (int): given reward by the action
            next_state (float): next state
            done (bool): if the episodic task done
        """
    
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.params.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.params.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.params.gamma)

    def act(self, state, i_episode=0, add_noise=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # add noise for exploration
        if add_noise:
            action_values = self.noise.sample(i_episode, action_values)
            
        return np.clip(action_values, -1, 1)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # compute and minimize the loss
        next_actions = self.actor_target(next_states)
        next_Qs = self.critic_target(next_states, next_actions).detach()
        
        target_Qs = rewards + gamma * next_Qs * (1 - dones)
        expected_Qs = self.critic_local(states, actions)
        
        loss_critic = F.mse_loss(expected_Qs, target_Qs)
        
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        # compute and maximize the policy
        pred_actions = self.actor_local(states)
        loss_actor = -self.critic_local(states, pred_actions).mean()
        
        self.actor_optimizer.zero_grad()    
        loss_actor.backward()
        self.actor_optimizer.step()

        # update target network
        self.soft_update(self.critic_local, self.critic_target, self.params.tau)                     
        self.soft_update(self.actor_local, self.actor_target, self.params.tau)
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)