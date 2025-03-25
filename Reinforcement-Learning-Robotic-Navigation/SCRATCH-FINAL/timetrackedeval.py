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

total_episodes = 20000

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
    start_time = time.time()  # Record start time
    
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

    # Record end time and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    metrics['execution_time'] = duration  # Add execution time to metrics
    
    log_message(f"Evaluation completed in {duration:.2f} seconds")
    
    # close log file, if opened
    if log_fp:
        log_fp.close()
   
    return metrics, rewards, lengths, successes, collisions, path_efficiencies if path_efficiencies else None, duration

def compare_agents(env, agents, agent_names, num_episodes=total_episodes, save_path=None, log_file=None, time_log_file=None):
    """
    Compare multiple agents on the same environment.
    
    Args:
        env: The environment.
        agents (list): List of agent objects.
        agent_names (list): List of agent names.
        num_episodes (int): Number of episodes for evaluation.
        save_path (str): Path to save comparison plots.
        log_file (str): Path to save comparison results.
        time_log_file (str): Path to save timing results.
    """
    assert len(agents) == len(agent_names), "Number of agents must match number of agent names"
    
    all_metrics = []
    execution_times = []
    total_start_time = time.time()  # Record start time for all agents

    # create results directory if it does not exist
    os.makedirs(os.path.dirname(log_file) if log_file else "results", exist_ok=True)
    
    # ensure the timing log directory exists
    if time_log_file:
        os.makedirs(os.path.dirname(time_log_file), exist_ok=True)

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

        metrics, _, _, _, _, _, duration = evaluate_agent(env, agent, num_episodes=num_episodes, log_file=agent_log_file)
        metrics['name'] = name
        all_metrics.append(metrics)
        execution_times.append((name, duration))
        
        log_message(f"Agent {name} completed evaluation in {duration:.2f} seconds")
    
    # Record total execution time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Write timing information to the dedicated timing log file
    if time_log_file:
        with open(time_log_file, 'w') as time_fp:
            time_fp.write("Agent Execution Times\n")
            time_fp.write("====================\n")
            for name, duration in execution_times:
                time_fp.write(f"{name}: {duration:.2f} seconds\n")
            time_fp.write("\n")
            time_fp.write(f"Total execution time for all agents: {total_duration:.2f} seconds\n")
    
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
    log_message("-" * 100)
    log_message(f"{'Agent':<15} {'Success Rate':<15} {'Mean Reward':<15} {'Mean Length':<15} {'Mean Collisions':<15} {'Execution Time (s)':<20}")
    log_message("-" * 100)
    for m, (name, duration) in zip(all_metrics, execution_times):
        log_message(f"{m['name']:<15} {m['success_rate']:<15.2f} {m['mean_reward']:<15.2f} {m['mean_length']:<15.2f} {m['mean_collisions']:<15.2f} {duration:<20.2f}")
    
    log_message(f"\nTotal execution time for all agents: {total_duration:.2f} seconds")
    
    # close log file, if opened
    if log_fp:
        log_fp.close()

    return all_metrics, execution_times, total_duration

if __name__ == "__main__":
    env = GridWorld()
    
    # create agents
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # compare multiple agents with timing
    agents = [
        DQNAgent(state_dim, action_dim),
        PolicyGradientAgent(state_dim, action_dim),
        PPOAgent(state_dim, action_dim)
    ]
    agent_names = ["DQN", "Policy Gradient", "PPO"]
    
    # Create results and times directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/times", exist_ok=True)
    
    # Compare agents and record timing information
    metrics, execution_times, total_duration = compare_agents(
        env, 
        agents, 
        agent_names, 
        num_episodes=total_episodes,
        save_path="results/agent_comparison.png",
        log_file="results/agent_comparison.txt",
        time_log_file="results/times/agent_timing.txt"
    )
    
    # Create a plot for timing comparison
    plt.figure(figsize=(10, 6))
    names = [name for name, _ in execution_times]
    durations = [duration for _, duration in execution_times]
    
    # Bar plot for execution times
    plt.bar(names, durations)
    plt.title('Agent Execution Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on top of each bar
    for i, duration in enumerate(durations):
        plt.text(i, duration + 0.1, f"{duration:.2f}s", ha='center')
    
    # Add the total time as text
    plt.figtext(0.5, 0.01, f"Total Time for All Agents: {total_duration:.2f} seconds", 
                ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for the text
    plt.savefig("results/times/timing_comparison.png")
    plt.show()
    
    print(f"Individual agent timing and total execution time saved to 'results/times/agent_timing.txt'")
    print(f"Timing comparison plot saved to 'results/times/timing_comparison.png'")