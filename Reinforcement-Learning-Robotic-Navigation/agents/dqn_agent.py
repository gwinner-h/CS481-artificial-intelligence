import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from .base_agent import BaseAgent
from utils.replay_buffer import ReplayBuffer
from utils.network_architectures import ConvNetwork, MLPNetwork

class DQNAgent(BaseAgent):
    """Agent implementing Deep Q-Network (DQN) algorithm."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10, device='cpu'):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim (tuple): Dimensions of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            epsilon_start (float): Starting value of epsilon for ε-greedy exploration.
            epsilon_end (float): Minimum value of epsilon.
            epsilon_decay (float): Decay rate of epsilon.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Batch size for training.
            target_update (int): Frequency of target network updates.
            device (str): Device to use for tensor operations.
        """
        super(DQNAgent, self).__init__(state_dim, action_dim, device)
        
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Determine whether to use CNN or MLP based on state dimensions
        # Grid-based state (2D)
        if isinstance(state_dim, tuple) and len(state_dim) > 1:
            # Use CNN for processing 2D grid
            self.q_network = ConvNetwork(state_dim, action_dim, hidden_dim).to(device)
            self.target_network = ConvNetwork(state_dim, action_dim, hidden_dim).to(device)
        else:  # Flat state (1D)
            # Use MLP for processing 1D state
            input_dim = state_dim[0] if isinstance(state_dim, tuple) else state_dim
            self.q_network = MLPNetwork(input_dim, action_dim, hidden_dim).to(device)
            self.target_network = MLPNetwork(input_dim, action_dim, hidden_dim).to(device)
        
        # Initialize target network with q_network weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode
        
        # Create optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize step counter for target network updates
        self.steps = 0
    
    def select_action(self, state, evaluation=False):
        """
        Select an action using ε-greedy policy.
        
        Args:
            state: The current state.
            evaluation (bool): Whether in evaluation mode.
            
        Returns:
            int: The selected action.
        """
        if (not evaluation) and (random.random() < self.epsilon):
            # Explore: select random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: select action with highest Q-value
            with torch.no_grad():
                state_tensor = self.preprocess_state(state)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state: The next state.
            done (bool): Whether the episode is done.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Increment step counter
        self.steps += 1
        
        # Update target network if needed
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update(self):
        """
        Update the Q-network using a batch from the replay buffer.
        
        Returns:
            float: The loss value.
        """
        # Check if enough samples are available
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Add channel dimension for CNN if using grid states
        if len(self.state_dim) > 1:
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)
        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """
        Save the agent's model to the specified path.
        
        Args:
            path (str): Path to save the model.
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """
        Load the agent's model from the specified path.
        
        Args:
            path (str): Path to load the model from.
        """
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']