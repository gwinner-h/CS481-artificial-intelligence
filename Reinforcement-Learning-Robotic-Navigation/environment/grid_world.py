# File: environment/grid_world.py
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class GridWorld(gym.Env):
    """
    A grid world environment for robot navigation tasks.
    
    The robot navigates in a 2D grid from a random start position to a goal,
    while avoiding obstacles. The agent receives a positive reward for reaching
    the goal and negative rewards for collisions and time penalties.
    """
    
    def __init__(self, width=10, height=10, obstacle_density=0.3, max_steps=100,
                 partial_observable=False, obs_radius=2):
        """
        Initialize the grid world environment.
        
        Args:
            width (int): Width of the grid.
            height (int): Height of the grid.
            obstacle_density (float): Density of obstacles in the grid.
            max_steps (int): Maximum number of steps per episode.
            partial_observable (bool): Whether the environment is partially observable.
            obs_radius (int): Observation radius if partially observable.
        """
        super(GridWorld, self).__init__()
        
        self.width = width
        self.height = height
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.partial_observable = partial_observable
        self.obs_radius = obs_radius
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: grid representation
        if self.partial_observable:
            # Partially observable: local view around the agent
            obs_width = 2 * self.obs_radius + 1
            obs_height = 2 * self.obs_radius + 1
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(obs_height, obs_width), dtype=np.float32
            )
        else:
            # Fully observable: full grid
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(self.height, self.width), dtype=np.float32
            )
        
        # Initialize grid
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0
        
        # Reset environment
        self.reset()
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            numpy.ndarray: Initial observation.
        """
        # Create empty grid
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Add obstacles
        num_obstacles = int(self.width * self.height * self.obstacle_density)
        obstacle_positions = np.random.choice(
            self.width * self.height, 
            size=num_obstacles, 
            replace=False
        )
        
        for pos in obstacle_positions:
            row = pos // self.width
            col = pos % self.width
            self.grid[row, col] = 1  # 1 represents obstacle
        
        # Place agent
        agent_placed = False
        while not agent_placed:
            row = np.random.randint(0, self.height)
            col = np.random.randint(0, self.width)
            if self.grid[row, col] == 0:  # Empty cell
                self.agent_pos = (row, col)
                self.grid[row, col] = 2  # 2 represents agent
                agent_placed = True
        
        # Place goal
        goal_placed = False
        while not goal_placed:
            row = np.random.randint(0, self.height)
            col = np.random.randint(0, self.width)
            if self.grid[row, col] == 0:  # Empty cell
                self.goal_pos = (row, col)
                self.grid[row, col] = 3  # 3 represents goal
                goal_placed = True
        
        # Reset step counter
        self.steps = 0
        
        # Return observation
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take (0=up, 1=right, 2=down, 3=left).
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Update step counter
        self.steps += 1
        
        # Calculate new position
        row, col = self.agent_pos
        new_row, new_col = row, col
        
        if action == 0:  # Up
            new_row = max(0, row - 1)
        elif action == 1:  # Right
            new_col = min(self.width - 1, col + 1)
        elif action == 2:  # Down
            new_row = min(self.height - 1, row + 1)
        elif action == 3:  # Left
            new_col = max(0, col - 1)
        
        # Check for collision
        collision = False
        if self.grid[new_row, new_col] == 1:  # Obstacle
            collision = True
            new_row, new_col = row, col  # Stay in place
        
        # Update grid
        self.grid[row, col] = 0  # Clear old position
        self.agent_pos = (new_row, new_col)
        
        # Check for goal
        reached_goal = (new_row, new_col) == self.goal_pos
        
        # Update agent position in grid (unless it's the goal)
        if not reached_goal:
            self.grid[new_row, new_col] = 2  # Agent
        
        # Calculate reward
        reward = 0
        if reached_goal:
            reward = 10.0  # Positive reward for reaching goal
        elif collision:
            reward = -1.0  # Negative reward for collision
        else:
            reward = -0.1  # Small negative reward for each step (time penalty)
        
        # Check if episode is done
        done = reached_goal or self.steps >= self.max_steps
        
        # Additional info
        info = {
            'success': reached_goal,
            'steps': self.steps,
            'collision': collision
        }
        
        # Return observation, reward, done, info
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """
        Get current observation of the environment.
        
        Returns:
            numpy.ndarray: Current observation.
        """
        if self.partial_observable:
            # Partially observable: local view around the agent
            row, col = self.agent_pos
            obs = np.zeros((2 * self.obs_radius + 1, 2 * self.obs_radius + 1), dtype=np.float32)
            
            for i in range(-self.obs_radius, self.obs_radius + 1):
                for j in range(-self.obs_radius, self.obs_radius + 1):
                    obs_row = i + self.obs_radius
                    obs_col = j + self.obs_radius
                    
                    grid_row = row + i
                    grid_col = col + j
                    
                    # Check if position is within grid
                    if 0 <= grid_row < self.height and 0 <= grid_col < self.width:
                        if (grid_row, grid_col) == self.goal_pos:
                            obs[obs_row, obs_col] = 0.5  # Goal
                        else:
                            obs[obs_row, obs_col] = self.grid[grid_row, grid_col] / 3.0
                    else:
                        obs[obs_row, obs_col] = 1.0  # Out of bounds treated as obstacle
            
            return obs
        else:
            # Fully observable: full grid
            obs = self.grid.copy().astype(np.float32) / 3.0
            
            # Highlight goal position
            goal_row, goal_col = self.goal_pos
            if self.agent_pos != self.goal_pos:  # Only if agent is not at goal
                obs[goal_row, goal_col] = 0.5
            
            return obs
    
    def render(self, mode='rgb_array'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode.
            
        Returns:
            numpy.ndarray: Grid representation for visualization.
        """
        if mode == 'human':
            plt.figure(figsize=(8, 8))
            plt.imshow(self.grid, cmap='viridis')
            plt.show()
        
        return self.grid.copy()
    
    def close(self):
        """Close the environment."""
        pass