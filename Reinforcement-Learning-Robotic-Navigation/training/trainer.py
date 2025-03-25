import numpy as np
import os
import time
import torch
from collections import deque
import matplotlib.pyplot as plt

class Trainer:
    """Trainer for reinforcement learning agents."""
    
    def __init__(self, env, agent, save_path='./checkpoints', eval_interval=100,
                 log_interval=10, save_interval=1000, render_interval=100,
                 renderer=None):
        """
        Initialize the trainer.
        
        Args:
            env: The environment.
            agent: The agent.
            save_path (str): Path to save checkpoints.
            eval_interval (int): Interval for evaluation.
            log_interval (int): Interval for logging.
            save_interval (int): Interval for saving checkpoints.
            render_interval (int): Interval for rendering.
            renderer: Renderer for visualization.
        """
        self.env = env
        self.agent = agent
        self.save_path = save_path
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.render_interval = render_interval
        self.renderer = renderer
        
        # Create save path if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []
        self.eval_rewards = []
    
    def train(self, num_episodes=1000):
        """
        Train the agent for a specified number of episodes.
        
        Args:
            num_episodes (int): Number of episodes to train for.
        """
        print(f"Starting training for {num_episodes} episodes...")
        start_time = time.time()
        
        # Initialize running metrics
        running_reward = 0
        running_length = 0
        running_success = 0
        
        for episode in range(1, num_episodes + 1):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            
            # Episode loop
            while not done:
                # Select action
                action = self.agent.select_action(state)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                
                # For DQN agent
                if hasattr(self.agent, 'store_transition'):
                    self.agent.store_transition(state, action, reward, next_state, done)
                
                # For Policy Gradient agent
                if hasattr(self.agent, 'store_reward'):
                    self.agent.store_reward(reward)
                
                # For PPO agent
                if hasattr(self.agent, 'rewards') and not hasattr(self.agent, 'store_transition') and not hasattr(self.agent, 'store_reward'):
                    self.agent.store_transition(reward, done)
                
                # Update state
                state = next_state
                
                # Update episode metrics
                episode_reward += reward
                episode_length += 1
                
                # Render if needed
                if self.renderer and episode % self.render_interval == 0:
                    self.renderer.render(self.env)
            
            # Update agent
            loss = self.agent.update()
            
            # Update metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            success = info.get('success', False)
            self.success_rate.append(float(success))
            
            # Update running metrics
            running_reward = 0.9 * running_reward + 0.1 * episode_reward
            running_length = 0.9 * running_length + 0.1 * episode_length
            running_success = 0.9 * running_success + 0.1 * float(success)
            
            # Log progress
            if episode % self.log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode}/{num_episodes} | " \
                      f"Reward: {episode_reward:.2f} | " \
                      f"Running Reward: {running_reward:.2f} | " \
                      f"Length: {episode_length} | " \
                      f"Success: {success} | " \
                      f"Running Success: {running_success:.2f} | " \
                      f"Time: {elapsed_time:.2f}s")
            
            # Evaluate agent
            if episode % self.eval_interval == 0:
                eval_rewards = self.evaluate(num_episodes=10)
                self.eval_rewards.append(np.mean(eval_rewards))
                print(f"Evaluation: Mean Reward: {np.mean(eval_rewards):.2f} | " \
                      f"Success Rate: {np.mean([r > 0 for r in eval_rewards]):.2f}")
            
            # Save checkpoint
            if episode % self.save_interval == 0:
                checkpoint_path = os.path.join(self.save_path, f"checkpoint_{episode}.pt")
                self.agent.save(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        # Final evaluation
        eval_rewards = self.evaluate(num_episodes=10)
        print(f"Final Evaluation: Mean Reward: {np.mean(eval_rewards):.2f} | " \
              f"Success Rate: {np.mean([r > 0 for r in eval_rewards]):.2f}")
        
        # Save final metrics plot
        self.plot_metrics(os.path.join(self.save_path, "training_metrics.png"))
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate the agent without exploration.
        
        Args:
            num_episodes (int): Number of episodes to evaluate for.
            render (bool): Whether to render during evaluation.
            
        Returns:
            list: Episode rewards.
        """
        eval_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, evaluation=True)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if render and self.renderer:
                    self.renderer.render(self.env)
            
            eval_rewards.append(episode_reward)
        
        return eval_rewards
    
    def plot_metrics(self, save_path=None):
        """
        Plot training metrics.
        
        Args:
            save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(15, 10))
        
        # Plot episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.plot(np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid'))
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_lengths)
        plt.plot(np.convolve(self.episode_lengths, np.ones(100)/100, mode='valid'))
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        
        # Plot success rate
        plt.subplot(2, 2, 3)
        plt.plot(np.convolve(self.success_rate, np.ones(100)/100, mode='valid'))
        plt.title('Success Rate (Moving Average)')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        
        # Plot evaluation rewards
        plt.subplot(2, 2, 4)
        plt.plot(np.arange(0, len(self.episode_rewards), self.eval_interval)[:len(self.eval_rewards)], 
                self.eval_rewards)
        plt.title('Evaluation Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.close()