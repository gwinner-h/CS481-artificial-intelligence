import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from environment.grid_world import GridWorld
from agents.dqn_agent import DQNAgent
from agents.policy_gradient_agent import PolicyGradientAgent
from agents.ppo_agent import PPOAgent
from visualization.renderer import Renderer
import time

total_episodes = 50000

def evaluate_agent(env, agent, num_episodes=total_episodes, render=False, renderer=None, fps=5, log_file=None):
    """
    Evaluate an agent on the environment.
   
    Args:
        env: The environment.
        agent: The agent.
        num_episodes (int): Number of episodes to evaluate for.
        render (bool): Whether to render during evaluation.
        renderer: Renderer for visualization.
        fps (int): Frames per second for rendering.
        log_file (str): Path to save the evaluation log.
       
    Returns:
        dict: Evaluation metrics.
    """
    rewards = []
    lengths = []
    successes = []
    collisions = []
    path_efficiencies = []

    # create results directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file) if log_file else "results", exist_ok=True)

    # open log file, if provided
    log_fp = open(log_file, 'w') if log_file else None

    def log_message(message):
        if log_fp:
            log_fp.write(message + "\n")
        else:
            print(message)
   
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        success = False
        collision_count = 0
        optimal_path_length = env.get_optimal_path_length() if hasattr(env, 'get_optimal_path_length') else None
       
        done = False
        while not done:
            action = agent.select_action(state, evaluation=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # check for collisions
            if 'collision' in info and info['collision']:
                collision_count += 1
                
            state = next_state
           
            if done and 'success' in info and info['success']:
                success = True
           
            if render and renderer:
                renderer.render(env)
                time.sleep(1 / fps)
       
        rewards.append(episode_reward)
        lengths.append(episode_length)
        successes.append(float(success))
        collisions.append(collision_count)
        
        # calculate path efficiency if optimal path is available
        if optimal_path_length is not None and episode_length > 0:
            efficiency = optimal_path_length / episode_length if success else 0
            path_efficiencies.append(efficiency)
       
        log_message(f"Episode {episode + 1}/{num_episodes} | " \
              f"Reward: {episode_reward:.2f} | " \
              f"Length: {episode_length} | " \
              f"Collisions: {collision_count} | " \
              f"Success: {success}")
   
    # calculate metrics
    metrics = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'success_rate': np.mean(successes),
        'mean_collisions': np.mean(collisions),
        'total_collisions': np.sum(collisions)
    }
    
    if path_efficiencies:
        metrics['mean_path_efficiency'] = np.mean(path_efficiencies)

    # close log file, if opened
    if log_fp:
        log_fp.close()
   
    return metrics, rewards, lengths, successes, collisions, path_efficiencies if path_efficiencies else None

def plot_evaluation_results(metrics, rewards, lengths, successes, collisions, path_efficiencies=None, save_path=None):
    """
    Plot evaluation results
   
    Args:
        metrics (dict): evaluation metrics
        rewards (list): episode rewards
        lengths (list): episode lengths
        successes (list): episode successes
        collisions (list): episode collision counts
        path_efficiencies (list): path efficiency scores
        save_path (str): path to save the plot
    """
    plt.figure(figsize=(15, 12))
   
    # plot episode rewards
    plt.subplot(3, 2, 1)
    plt.hist(rewards, bins=20)
    plt.axvline(metrics['mean_reward'], color='r', linestyle='--',
                label=f"Mean: {metrics['mean_reward']:.2f}")
    plt.title('Episode Rewards')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.legend()
   
    # Plot episode lengths
    plt.subplot(3, 2, 2)
    plt.hist(lengths, bins=20)
    plt.axvline(metrics['mean_length'], color='r', linestyle='--',
                label=f"Mean: {metrics['mean_length']:.2f}")
    plt.title('Episode Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.legend()
   
    # Plot success rate
    plt.subplot(3, 2, 3)
    labels = ['Failure', 'Success']
    counts = [len(successes) - sum(successes), sum(successes)]
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title(f"Success Rate: {metrics['success_rate']:.2f}")
    
    # Plot collision histogram
    plt.subplot(3, 2, 4)
    plt.hist(collisions, bins=min(20, max(collisions) + 1))
    plt.axvline(metrics['mean_collisions'], color='r', linestyle='--',
                label=f"Mean: {metrics['mean_collisions']:.2f}")
    plt.title('Collisions per Episode')
    plt.xlabel('Number of Collisions')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot path efficiency if available
    if path_efficiencies is not None:
        plt.subplot(3, 2, 5)
        plt.hist(path_efficiencies, bins=20)
        plt.axvline(metrics['mean_path_efficiency'], color='r', linestyle='--',
                label=f"Mean: {metrics['mean_path_efficiency']:.2f}")
        plt.title('Path Efficiency')
        plt.xlabel('Efficiency (Optimal/Actual)')
        plt.ylabel('Count')
        plt.legend()
   
    # Plot metrics table
    plt.subplot(3, 2, 6)
    plt.axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Mean Reward', f"{metrics['mean_reward']:.2f}"],
        ['Std Reward', f"{metrics['std_reward']:.2f}"],
        ['Min Reward', f"{metrics['min_reward']:.2f}"],
        ['Max Reward', f"{metrics['max_reward']:.2f}"],
        ['Mean Length', f"{metrics['mean_length']:.2f}"],
        ['Std Length', f"{metrics['std_length']:.2f}"],
        ['Success Rate', f"{metrics['success_rate']:.2f}"],
        ['Mean Collisions', f"{metrics['mean_collisions']:.2f}"],
        ['Total Collisions', f"{metrics['total_collisions']}"]
    ]
    
    if path_efficiencies is not None:
        table_data.append(['Mean Path Efficiency', f"{metrics['mean_path_efficiency']:.2f}"])
        
    plt.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.4, 0.4])
    plt.title('Evaluation Metrics')
   
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()

def compare_agents(env, agents, agent_names, num_episodes=total_episodes, save_path=None, log_file=None):
    """
    Compare multiple agents on the same environment.
    
    Args:
        env: The environment.
        agents (list): List of agent objects.
        agent_names (list): List of agent names.
        num_episodes (int): Number of episodes for evaluation.
        save_path (str): Path to save comparison plots.
        log_file (str): Path to save comparison results.
    """
    assert len(agents) == len(agent_names), "Number of agents must match number of agent names"
    
    all_metrics = []

    # create results directory if it does not exist
    os.makedirs(os.path.dirname(log_file) if log_file else "results", exist_ok=True)

    # open the log file, if provided
    log_fp = open(log_file, 'w') if log_file else None

    def log_message(message):
        if log_fp:
            log_fp.write(message + "\n")
        else:
            print(message)
    
    for agent, name in zip(agents, agent_names):
        log_message(f"\nEvaluating agent: {name}")

        # create individual agent log file
        agent_log_file = None
        if log_file:
            agent_log_dir = os.path.dirname(log_file)
            agent_log_name = f"{os.path.splitext(os.path.basename(log_file))[0]}_{name}.txt"
            agent_log_file = os.path.join(agent_log_dir, agent_log_name)

        metrics, _, _, _, _, _ = evaluate_agent(env, agent, num_episodes=num_episodes, log_file=agent_log_file)
        metrics['name'] = name
        all_metrics.append(metrics)
    
    # create comparison plots
    plt.figure(figsize=(15, 10))
    
    # bar plot for success rate
    plt.subplot(2, 2, 1)
    names = [m['name'] for m in all_metrics]
    success_rates = [m['success_rate'] for m in all_metrics]
    plt.bar(names, success_rates)
    plt.title('Success Rate Comparison')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    
    # bar plot for mean reward
    plt.subplot(2, 2, 2)
    mean_rewards = [m['mean_reward'] for m in all_metrics]
    plt.bar(names, mean_rewards)
    plt.title('Mean Reward Comparison')
    plt.ylabel('Mean Reward')
    
    # bar plot for mean episode length
    plt.subplot(2, 2, 3)
    mean_lengths = [m['mean_length'] for m in all_metrics]
    plt.bar(names, mean_lengths)
    plt.title('Mean Episode Length Comparison')
    plt.ylabel('Steps')
    
    # bar plot for mean collisions
    plt.subplot(2, 2, 4)
    mean_collisions = [m['mean_collisions'] for m in all_metrics]
    plt.bar(names, mean_collisions)
    plt.title('Mean Collisions Comparison')
    plt.ylabel('Collisions')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()
    
    # write comparison table to log file
    log_message("\nAgent Comparison Summary:")
    log_message("-" * 80)
    log_message(f"{'Agent':<15} {'Success Rate':<15} {'Mean Reward':<15} {'Mean Length':<15} {'Mean Collisions':<15}")
    log_message("-" * 80)
    for m in all_metrics:
        log_message(f"{m['name']:<15} {m['success_rate']:<15.2f} {m['mean_reward']:<15.2f} {m['mean_length']:<15.2f} {m['mean_collisions']:<15.2f}")
    
    # close log file, if opened
    if log_fp:
        log_fp.close()

    return all_metrics

def evaluate_hyperparameters(env_class, agent_class, hyperparams_list, param_names, num_episodes=total_episodes, save_path=None):
    """
    Evaluate the same agent with different hyperparameters.
    
    Args:
        env_class: The environment class.
        agent_class: The agent class.
        hyperparams_list (list): List of hyperparameter dictionaries.
        param_names (list): List of parameter names to display in results.
        num_episodes (int): Number of episodes for evaluation.
        save_path (str): Path to save comparison plots.
    """
    all_metrics = []
    param_labels = []
    
    for i, hyperparams in enumerate(hyperparams_list):
        # Create environment and agent with current hyperparameters
        env = env_class()
        agent = agent_class(env.observation_space, env.action_space, **hyperparams)
        
        # Create label for this parameter set
        label = ", ".join([f"{name}={hyperparams[name]}" for name in param_names])
        param_labels.append(label)
        
        print(f"\nEvaluating hyperparameters set {i+1}/{len(hyperparams_list)}:")
        print(label)
        
        metrics, _, _, _, _, _ = evaluate_agent(env, agent, num_episodes=num_episodes)
        metrics['label'] = label
        all_metrics.append(metrics)
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Bar plot for success rate
    plt.subplot(2, 2, 1)
    success_rates = [m['success_rate'] for m in all_metrics]
    plt.bar(range(len(param_labels)), success_rates)
    plt.xticks(range(len(param_labels)), range(1, len(param_labels) + 1))
    plt.title('Success Rate by Hyperparameters')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    
    # Bar plot for mean reward
    plt.subplot(2, 2, 2)
    mean_rewards = [m['mean_reward'] for m in all_metrics]
    plt.bar(range(len(param_labels)), mean_rewards)
    plt.xticks(range(len(param_labels)), range(1, len(param_labels) + 1))
    plt.title('Mean Reward by Hyperparameters')
    plt.ylabel('Mean Reward')
    
    # Bar plot for mean episode length
    plt.subplot(2, 2, 3)
    mean_lengths = [m['mean_length'] for m in all_metrics]
    plt.bar(range(len(param_labels)), mean_lengths)
    plt.xticks(range(len(param_labels)), range(1, len(param_labels) + 1))
    plt.title('Mean Episode Length by Hyperparameters')
    plt.ylabel('Steps')
    
    # Bar plot for mean collisions
    plt.subplot(2, 2, 4)
    mean_collisions = [m['mean_collisions'] for m in all_metrics]
    plt.bar(range(len(param_labels)), mean_collisions)
    plt.xticks(range(len(param_labels)), range(1, len(param_labels) + 1))
    plt.title('Mean Collisions by Hyperparameters')
    plt.ylabel('Collisions')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()
    
    # Print comparison table
    print("\nHyperparameter Comparison Summary:")
    print("-" * 100)
    header = f"{'Set':<5} {'Success Rate':<15} {'Mean Reward':<15} {'Mean Length':<15} {'Mean Collisions':<15} {'Parameters'}"
    print(header)
    print("-" * 100)
    for i, m in enumerate(all_metrics):
        print(f"{i+1:<5} {m['success_rate']:<15.2f} {m['mean_reward']:<15.2f} {m['mean_length']:<15.2f} {m['mean_collisions']:<15.2f} {m['label']}")
    
    return all_metrics

def evaluate_training_progress(env, agent, checkpoint_paths, checkpoint_names, num_episodes=total_episodes, save_path=None):
    """
    Evaluate agent checkpoints to measure training progress.
    
    Args:
        env: The environment.
        agent: The agent (base model for loading checkpoints).
        checkpoint_paths (list): List of paths to model checkpoints.
        checkpoint_names (list): List of names for the checkpoints (e.g., "10k steps").
        num_episodes (int): Number of episodes for evaluation.
        save_path (str): Path to save comparison plots.
    """
    all_metrics = []
    
    for path, name in zip(checkpoint_paths, checkpoint_names):
        print(f"\nEvaluating checkpoint: {name}")
        
        # Load checkpoint
        agent.load(path)
        
        metrics, _, _, _, _, _ = evaluate_agent(env, agent, num_episodes=num_episodes)
        metrics['name'] = name
        all_metrics.append(metrics)
    
    # Create progress plots
    plt.figure(figsize=(15, 10))
    
    names = [m['name'] for m in all_metrics]
    
    # Line plot for success rate
    plt.subplot(2, 2, 1)
    success_rates = [m['success_rate'] for m in all_metrics]
    plt.plot(range(len(names)), success_rates, 'o-')
    plt.xticks(range(len(names)), names)
    plt.title('Success Rate Over Training')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    
    # Line plot for mean reward
    plt.subplot(2, 2, 2)
    mean_rewards = [m['mean_reward'] for m in all_metrics]
    plt.plot(range(len(names)), mean_rewards, 'o-')
    plt.xticks(range(len(names)), names)
    plt.title('Mean Reward Over Training')
    plt.ylabel('Mean Reward')
    
    # Line plot for mean episode length
    plt.subplot(2, 2, 3)
    mean_lengths = [m['mean_length'] for m in all_metrics]
    plt.plot(range(len(names)), mean_lengths, 'o-')
    plt.xticks(range(len(names)), names)
    plt.title('Mean Episode Length Over Training')
    plt.ylabel('Steps')
    
    # Line plot for mean collisions
    plt.subplot(2, 2, 4)
    mean_collisions = [m['mean_collisions'] for m in all_metrics]
    plt.plot(range(len(names)), mean_collisions, 'o-')
    plt.xticks(range(len(names)), names)
    plt.title('Mean Collisions Over Training')
    plt.ylabel('Collisions')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()
    
    # Print progress table
    print("\nTraining Progress Summary:")
    print("-" * 80)
    print(f"{'Checkpoint':<15} {'Success Rate':<15} {'Mean Reward':<15} {'Mean Length':<15} {'Mean Collisions':<15}")
    print("-" * 80)
    for m in all_metrics:
        print(f"{m['name']:<15} {m['success_rate']:<15.2f} {m['mean_reward']:<15.2f} {m['mean_length']:<15.2f} {m['mean_collisions']:<15.2f}")
    
    return all_metrics

def evaluate_environment_variations(env_class, env_configs, config_names, agent, num_episodes=total_episodes, save_path=None):
    """
    Evaluate the same agent across different environment configurations.
    
    Args:
        env_class: The environment class.
        env_configs (list): List of environment configuration dictionaries.
        config_names (list): Names for each environment configuration.
        agent: The agent to evaluate.
        num_episodes (int): Number of episodes for evaluation.
        save_path (str): Path to save comparison plots.
    """
    all_metrics = []
    
    for config, name in zip(env_configs, config_names):
        print(f"\nEvaluating environment: {name}")
        
        # Create environment with current configuration
        env = env_class(**config)
        
        metrics, _, _, _, _, _ = evaluate_agent(env, agent, num_episodes=num_episodes)
        metrics['name'] = name
        all_metrics.append(metrics)
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    names = [m['name'] for m in all_metrics]
    
    # Bar plot for success rate
    plt.subplot(2, 2, 1)
    success_rates = [m['success_rate'] for m in all_metrics]
    plt.bar(names, success_rates)
    plt.title('Success Rate by Environment')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Bar plot for mean reward
    plt.subplot(2, 2, 2)
    mean_rewards = [m['mean_reward'] for m in all_metrics]
    plt.bar(names, mean_rewards)
    plt.title('Mean Reward by Environment')
    plt.ylabel('Mean Reward')
    plt.xticks(rotation=45, ha='right')
    
    # Bar plot for mean episode length
    plt.subplot(2, 2, 3)
    mean_lengths = [m['mean_length'] for m in all_metrics]
    plt.bar(names, mean_lengths)
    plt.title('Mean Episode Length by Environment')
    plt.ylabel('Steps')
    plt.xticks(rotation=45, ha='right')
    
    # Bar plot for mean collisions
    plt.subplot(2, 2, 4)
    mean_collisions = [m['mean_collisions'] for m in all_metrics]
    plt.bar(names, mean_collisions)
    plt.title('Mean Collisions by Environment')
    plt.ylabel('Collisions')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()
    
    # Print comparison table
    print("\nEnvironment Comparison Summary:")
    print("-" * 80)
    print(f"{'Environment':<20} {'Success Rate':<15} {'Mean Reward':<15} {'Mean Length':<15} {'Mean Collisions':<15}")
    print("-" * 80)
    for m in all_metrics:
        print(f"{m['name']:<20} {m['success_rate']:<15.2f} {m['mean_reward']:<15.2f} {m['mean_length']:<15.2f} {m['mean_collisions']:<15.2f}")
    
    return all_metrics

if __name__ == "__main__":

    env = GridWorld()
    
    # create an agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    # evaluate a single agent
    metrics, rewards, lengths, successes, collisions, path_efficiencies = evaluate_agent(
        env, agent, num_episodes=total_episodes, render=True, log_file="results/evaluation_log.txt")
    
    # plot the results
    plot_evaluation_results(metrics, rewards, lengths, successes, collisions, 
                           path_efficiencies, save_path="results/evaluation.png")
    
    # compare multiple agents
    agents = [
        DQNAgent(state_dim, action_dim),
        PolicyGradientAgent(state_dim, action_dim),
        PPOAgent(state_dim, action_dim)
    ]
    agent_names = ["DQN", "Policy Gradient", "PPO"]
    
    compare_metrics = compare_agents(env, agents, agent_names, 
                                    num_episodes=total_episodes,
                                    save_path="results/agent_comparison.png",
                                    log_file="results/agent_comparison.txt")