import torch
import torch.nn as nn

class ConvNetwork(nn.Module):
    """Convolutional neural network for processing grid-based states."""
    
    def __init__(self, input_shape, output_dim, hidden_dim=128):
        """
        Initialize the convolutional network.
        
        Args:
            input_shape (tuple): Shape of the input state (height, width).
            output_dim (int): Dimension of the output.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(ConvNetwork, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        with torch.no_grad():
            sample = torch.zeros(1, 1, *input_shape)
            flat_size = self.conv(sample).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        # Add channel dimension if not present
        if len(x.shape) == 3:  # (batch_size, height, width)
            x = x.unsqueeze(1)  # (batch_size, 1, height, width)
        
        x = self.conv(x)
        return self.fc(x)


class MLPNetwork(nn.Module):
    """Multi-layer perceptron for processing flat states."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Initialize the MLP network.
        
        Args:
            input_dim (int): Dimension of the input state.
            output_dim (int): Dimension of the output.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(MLPNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.fc(x)