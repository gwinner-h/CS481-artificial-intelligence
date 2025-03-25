import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from utils.network_architectures import ConvNetwork, MLPNetwork

class PolicyNet(nn.Module):
    """Neural network for policy gradient methods."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the policy network.
        
        Args:
            state_dim (int or tuple): Dimensions of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(PolicyNet, self).__init__()
        
        # Determine whether to use CNN or MLP based on state dimensions
        if isinstance(state_dim, tuple) and len(state_dim) > 1:  # Grid-based state (2D)
            # Use CNN for processing 2D grid
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # Calculate flattened size
            with torch.no_grad():
                sample = torch.zeros(1, 1, *state_dim)
                flat_size = self.features(sample).shape[1]
            
            self.fc = nn.Sequential(
                nn.Linear(flat_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=1)
            )
        else:  # Flat state (1D)
            # Use MLP for processing 1D state
            input_dim = state_dim[0] if isinstance(state_dim, tuple) else state_dim
            self.features = nn.Identity()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=1)
            )
    
    def forward(self, x):
        """Forward pass through the network."""
        # Add channel dimension if needed
        if len(x.shape) == 3:  # (batch_size, height, width)
            x = x.unsqueeze(1)  # (batch_size, 1, height, width)
        
        x = self.features(x)
        return self.fc(x)

class PolicyGradientAgent(BaseAgent):
    """Agent implementing Policy Gradient (REINFORCE) algorithm."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, 
                 gamma=0.99, device='cpu'):
        """
        Initialize the policy gradient agent.
        
        Args:
            state_dim (tuple): Dimensions of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            device (str): Device to use for tensor operations.
        """
        super(PolicyGradientAgent, self).__init__(state_dim, action_dim, device)
        
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        
        # Create policy network
        self.policy = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize episode memory
        self.reset_episode()
    
    def reset_episode(self):
        """Reset the episode memory."""
        self.log_probs = []
        self.rewards = []
        self.entropies = []
    
    def select_action(self, state, evaluation=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: The current state.
            evaluation (bool): Whether in evaluation mode.
            
        Returns:
            int: The selected action.
        """
        state_tensor = self.preprocess_state(state)
        
        # Get action probabilities
        probs = self.policy(state_tensor)
        
        # Create categorical distribution
        m = torch.distributions.Categorical(probs)
        
        if evaluation:
            # In evaluation mode, select the action with highest probability
            action = torch.argmax(probs).item()
        else:
            # In training mode, sample from the distribution
            action = m.sample()
            
            # Store log probability and entropy for training
            self.log_probs.append(m.log_prob(action))
            self.entropies.append(m.entropy())
            
            action = action.item()
        
        return action
    
    def store_reward(self, reward):
        """
        Store a reward in the episode memory.
        
        Args:
            reward (float): The reward received.
        """
        self.rewards.append(reward)
    
    def update(self):
        """
        Update the policy using collected trajectories.
        
        Returns:
            float: The loss value.
        """
        # Check if there are any rewards to learn from
        if len(self.rewards) == 0:
            return 0.0
        
        # Calculate discounted rewards
        R = 0
        discounted_rewards = []
        
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        # Convert to tensor
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        # Convert log probs and entropies to tensors
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)
        
        # Calculate loss
        policy_loss = -torch.sum(log_probs * discounted_rewards)
        entropy_loss = -torch.sum(entropies) * 0.01  # Regularization term
        
        loss = policy_loss + entropy_loss
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Reset episode memory
        self.reset_episode()
        
        return loss.item()
    
    def save(self, path):
        """
        Save the agent's model to the specified path.
        
        Args:
            path (str): Path to save the model.
        """
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """
        Load the agent's model from the specified path.
        
        Args:
            path (str): Path to load the model from.
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])