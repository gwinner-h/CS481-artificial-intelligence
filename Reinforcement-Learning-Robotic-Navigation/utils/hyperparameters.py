import json
import os

class Hyperparameters:
    """Class for managing hyperparameters."""
    
    def __init__(self, algorithm='dqn'):
        """
        Initialize default hyperparameters.
        
        Args:
            algorithm (str): Algorithm name ('dqn', 'pg', or 'ppo').
        """
        # Environment parameters
        self.env_params = {
            'width': 10,
            'height': 10,
            'obstacle_density': 0.3,
            'max_steps': 100,
            'partial_observable': False,
            'obs_radius': 2
        }
        
        # Common agent parameters
        self.agent_params = {
            'hidden_dim': 128,
            'lr': 0.001,
            'gamma': 0.99
        }
        
        # Algorithm-specific parameters
        if algorithm == 'dqn':
            self.agent_params.update({
                'epsilon_start': 1.0,
                'epsilon_end': 0.1,
                'epsilon_decay': 0.995,
                'buffer_size': 10000,
                'batch_size': 64,
                'target_update': 10
            })
        elif algorithm == 'pg':
            pass  # No additional parameters
        elif algorithm == 'ppo':
            self.agent_params.update({
                'clip_ratio': 0.2,
                'n_epochs': 10,
                'batch_size': 64,
                'value_coef': 0.5,
                'entropy_coef': 0.01,
                'lam': 0.95
            })
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Training parameters
        self.train_params = {
            'num_episodes': 1000,
            'eval_interval': 100,
            'log_interval': 10,
            'save_interval': 1000,
            'render_interval': 100
        }
    
    def save(self, path):
        """
        Save hyperparameters to a JSON file.
        
        Args:
            path (str): Path to save the hyperparameters.
        """
        params = {
            'env_params': self.env_params,
            'agent_params': self.agent_params,
            'train_params': self.train_params
        }
        
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)
    
    def load(self, path):
        """
        Load hyperparameters from a JSON file.
        
        Args:
            path (str): Path to load the hyperparameters from.
        """
        with open(path, 'r') as f:
            params = json.load(f)
        
        self.env_params = params['env_params']
        self.agent_params = params['agent_params']
        self.train_params = params['train_params']
    
    def get_env_params(self):
        """Return environment parameters."""
        return self.env_params.copy()
    
    def get_agent_params(self):
        """Return agent parameters."""
        return self.agent_params.copy()
    
    def get_train_params(self):
        """Return training parameters."""
        return self.train_params.copy()