import torch
import numpy as np

class BaseAgent:
    """Base class for all reinforcement learning agents."""
    
    def __init__(self, state_dim, action_dim, device='cpu'):
        """
        Initialize the base agent.
        
        Args:
            state_dim (tuple): Dimensions of the state space.
            action_dim (int): Dimension of the action space.
            device (str): Device to use for tensor operations.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
    
    def select_action(self, state, evaluation=False):
        """
        Select an action based on the current state.
        
        Args:
            state: The current state.
            evaluation (bool): Whether in evaluation mode.
            
        Returns:
            int: The selected action.
        """
        raise NotImplementedError("Subclasses must implement select_action method")
    
    def update(self):
        """Update the agent's policy."""
        raise NotImplementedError("Subclasses must implement update method")
    
    def save(self, path):
        """
        Save the agent's model to the specified path.
        
        Args:
            path (str): Path to save the model.
        """
        raise NotImplementedError("Subclasses must implement save method")
    
    def load(self, path):
        """
        Load the agent's model from the specified path.
        
        Args:
            path (str): Path to load the model from.
        """
        raise NotImplementedError("Subclasses must implement load method")
    
    def preprocess_state(self, state):
        """
        Preprocess the state for network input.
        
        Args:
            state: The raw state.
            
        Returns:
            torch.Tensor: The preprocessed state.
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Add batch dimension if needed
        if isinstance(self.state_dim, (tuple, list)) and len(state.shape) == len(self.state_dim):
            state = state.unsqueeze(0)
        
        return state