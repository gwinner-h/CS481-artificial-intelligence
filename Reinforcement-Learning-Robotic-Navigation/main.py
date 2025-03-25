import argparse
import torch
import numpy as np
import random
import os

from environment.grid_world import GridWorld
from agents.dqn_agent import DQNAgent
from agents.policy_gradient_agent import PolicyGradientAgent
from agents.ppo_agent import PPOAgent
from utils.hyperparameters import Hyperparameters
from training.trainer import Trainer
from visualization.renderer import Renderer

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Robot Navigation RL')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['dqn', 'pg', 'ppo'],
                        help='RL algorithm to use')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, default=None, help='Configuration file')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load hyperparameters
    hyperparams = Hyperparameters(algorithm=args.algorithm)
    if args.config:
        hyperparams.load(args.config)
    
    # Create environment
    env = GridWorld(**hyperparams.get_env_params())
    
    # Determine state and action dimensions
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    # Create agent
    if args.algorithm == 'dqn':
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=args.device,
            **hyperparams.get_agent_params()
        )
    elif args.algorithm == 'pg':
        agent = PolicyGradientAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=args.device,
            **hyperparams.get_agent_params()
        )
    elif args.algorithm == 'ppo':
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=args.device,
            **hyperparams.get_agent_params()
        )
    
    # Load checkpoint if available
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        agent.load(args.checkpoint)
    
    # Create renderer if needed
    renderer = Renderer(env) if args.render else None
    
    # Evaluation mode
    if args.eval:
        from evaluate import evaluate_agent, plot_evaluation_results
        
        print(f"Evaluating {args.algorithm.upper()} agent...")
        metrics, rewards, lengths, successes = evaluate_agent(
            env=env,
            agent=agent,
            num_episodes=100,
            render=args.render,
            renderer=renderer
        )
        
        print(f"Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        plot_evaluation_results(
            metrics=metrics,
            rewards=rewards,
            lengths=lengths,
            successes=successes,
            save_path=f"results/{args.algorithm}_evaluation.png"
        )
    # Training mode
    else:
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Create trainer
        trainer = Trainer(
            env=env,
            agent=agent,
            save_path=f"results/{args.algorithm}_checkpoints",
            renderer=renderer,
            **hyperparams.get_train_params()
        )
        
        # Save hyperparameters
        hyperparams.save(f"results/{args.algorithm}_hyperparams.json")
        
        # Start training
        trainer.train(num_episodes=hyperparams.train_params['num_episodes'])
        # trainer.train(num_episodes=100)

if __name__ == "__main__":
    main()