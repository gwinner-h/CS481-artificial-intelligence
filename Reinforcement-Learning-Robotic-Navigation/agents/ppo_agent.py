import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent

class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the actor-critic network.
        
        Args:
            state_dim (int or tuple): Dimensions of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(ActorCritic, self).__init__()
        
        # For image-like states (grid)
        if isinstance(state_dim, tuple) and len(state_dim) > 1:
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # Calculate the size after flattening
            with torch.no_grad():
                sample = torch.zeros(1, 1, *state_dim)
                flat_size = self.features(sample).shape[1]
            
            # Shared network up to this point
            self.fc_shared = nn.Sequential(
                nn.Linear(flat_size, hidden_dim),
                nn.ReLU()
            )
        else:
            # For flat states
            self.features = nn.Identity()
            input_dim = state_dim[0] if isinstance(state_dim, tuple) else state_dim
            self.fc_shared = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """Forward pass through the network."""
        if len(x.shape) > 3:  # If input is a batch of 2D grids
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, height, width)
        
        x = self.features(x)
        x = self.fc_shared(x)
        
        # Policy (actor)
        action_probs = self.softmax(self.actor(x))
        
        # Value (critic)
        value = self.critic(x)
        
        return action_probs, value
    
    def get_action_probs(self, x):
        """Get action probabilities without value."""
        if len(x.shape) > 3:  # If input is a batch of 2D grids
            # Continuation of agents/ppo_agent.py
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, height, width)
        
        x = self.features(x)
        x = self.fc_shared(x)
        return self.softmax(self.actor(x))
    
    def get_value(self, x):
        """Get state value without action probabilities."""
        if len(x.shape) > 3:  # If input is a batch of 2D grids
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, height, width)
        
        x = self.features(x)
        x = self.fc_shared(x)
        return self.critic(x)

class PPOAgent(BaseAgent):
    """Agent implementing Proximal Policy Optimization (PPO)."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.0003,
                 gamma=0.99, clip_ratio=0.2, n_epochs=10, batch_size=64,
                 value_coef=0.5, entropy_coef=0.01, lam=0.95, device='cpu'):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim (tuple): Dimensions of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            clip_ratio (float): PPO clipping parameter.
            n_epochs (int): Number of epochs to optimize on each batch.
            batch_size (int): Batch size for training.
            value_coef (float): Value loss coefficient.
            entropy_coef (float): Entropy coefficient for exploration.
            lam (float): GAE lambda parameter.
            device (str): Device to use for tensor operations.
        """
        super(PPOAgent, self).__init__(state_dim, action_dim, device)
        
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.lam = lam
        
        # Create actor-critic network
        self.ac_network = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr)
        
        # Initialize trajectory buffer
        self.reset_trajectory()
    
    def reset_trajectory(self):
        """Reset the trajectory buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state, evaluation=False):
        """
        Select an action based on the current policy.
        
        Args:
            state: The current state.
            evaluation (bool): Whether in evaluation mode.
            
        Returns:
            int: The selected action.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.ac_network(state_tensor)
            
            if evaluation:
                # In evaluation mode, select the action with highest probability
                action = torch.argmax(action_probs).item()
            else:
                # In training mode, sample from the probability distribution
                m = torch.distributions.Categorical(action_probs)
                action = m.sample().item()
                
                self.states.append(state)
                self.actions.append(action)
                self.values.append(value.item())
                self.log_probs.append(m.log_prob(torch.tensor(action)).item())
            
            return action
    
    def store_transition(self, reward, done):
        """
        Store a transition in the trajectory buffer.
        
        Args:
            reward (float): The reward received.
            done (bool): Whether the episode is done.
        """
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_advantages(self):
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        # Calculate GAE advantages
        advantages = []
        gae = 0
        
        with torch.no_grad():
            # Get the last state value if the episode is not done
            if len(self.states) > 0 and not self.dones[-1]:
                state_tensor = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
                _, next_value = self.ac_network(state_tensor)
                next_value = next_value.item()
            else:
                next_value = 0
        
        # Iterate in reverse order
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = next_value * next_non_terminal
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1] * next_non_terminal
            
            delta = self.rewards[t] + self.gamma * next_value - self.values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages.insert(0, gae)
        
        # Convert to tensor
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def update(self):
        """Update the policy and value function using PPO."""
        # Check if there are any transitions to learn from
        if len(self.rewards) == 0:
            return
        
        # Compute advantages and returns
        advantages = self.compute_advantages()
        returns = advantages + torch.tensor(self.values, dtype=torch.float32).to(self.device)
        
        # Convert trajectory to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Training loop
        total_loss = 0
        
        for _ in range(self.n_epochs):
            # Get current action probabilities and state values
            action_probs, values = self.ac_network(states)
            values = values.squeeze()
            
            # Create categorical distribution
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            
            # Calculate total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Reset trajectory buffer
        self.reset_trajectory()
        
        return total_loss / self.n_epochs
    
    def save(self, path):
        """
        Save the agent's model to the specified path.
        
        Args:
            path (str): Path to save the model.
        """
        torch.save({
            'ac_network': self.ac_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """
        Load the agent's model from the specified path.
        
        Args:
            path (str): Path to load the model from.
        """
        checkpoint = torch.load(path)
        self.ac_network.load_state_dict(checkpoint['ac_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])