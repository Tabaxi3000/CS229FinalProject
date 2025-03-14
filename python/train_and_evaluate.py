#!/usr/bin/env python3

import os
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
from typing import List, Dict, Any
plt.ion()  # Turn on interactive mode

# Import our models and environment
from dqn import DQNAgent, DQNetwork
from reinforce import REINFORCEAgent, PolicyNetwork
from mcts import MCTSAgent, PolicyValueNetwork
from gin_rummy_env import GinRummyEnv
from improved_training import ImprovedDQNAgent, ImprovedREINFORCEAgent, train_dqn, train_reinforce, self_play_training
from rules_based_agent import RulesBasedAgent
from improved_gin_rummy_env import ImprovedGinRummyEnv

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def evaluate_models(num_games=50, verbose=False):
    """Evaluate all models against random and against each other"""
    print("\n=== Evaluating Models ===\n")
    
    # Create environment
    env = GinRummyEnv()
    
    # Create agents
    dqn_agent = ImprovedDQNAgent()
    reinforce_agent = ImprovedREINFORCEAgent()
    
    # Load models if available
    dqn_agent.load("models/improved_dqn_final.pt")
    reinforce_agent.load("models/improved_reinforce_final.pt")
    
    # Create a random agent
    class RandomAgent:
        def select_action(self, state, eval_mode=False):
            valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_actions, int):
                valid_actions = [valid_actions]
            return random.choice(valid_actions)
    
    random_agent = RandomAgent()
    
    # Evaluate DQN vs Random
    print("\nEvaluating DQN vs Random...")
    dqn_vs_random_results = evaluate_agent_vs_agent(dqn_agent, random_agent, env, num_games, verbose)
    
    # Evaluate REINFORCE vs Random
    print("\nEvaluating REINFORCE vs Random...")
    reinforce_vs_random_results = evaluate_agent_vs_agent(reinforce_agent, random_agent, env, num_games, verbose)
    
    # Evaluate DQN vs REINFORCE
    print("\nEvaluating DQN vs REINFORCE...")
    dqn_vs_reinforce_results = evaluate_agent_vs_agent(dqn_agent, reinforce_agent, env, num_games, verbose)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"DQN vs Random: {dqn_vs_random_results}")
    print(f"REINFORCE vs Random: {reinforce_vs_random_results}")
    print(f"DQN vs REINFORCE: {dqn_vs_reinforce_results}")

def evaluate_agent_vs_agent(agent1, agent2, env, num_games=10, verbose=False):
    """Evaluate agent1 vs agent2"""
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for game in tqdm(range(num_games), desc="Games"):
        state = env.reset()
        done = False
        turn = 0
        
        while not done and turn < 100:  # Limit to 100 turns to prevent infinite games
            if verbose and turn % 10 == 0:
                print(f"\nGame {game+1}, Turn {turn+1}")
                env.print_state()
            
            # Agent 1's turn
            if env.current_player == 0:
                action = agent1.select_action(state, eval_mode=True)
                if verbose:
                    print(f"Agent 1 took action {action}")
            # Agent 2's turn
            else:
                action = agent2.select_action(state, eval_mode=True)
                if verbose:
                    print(f"Agent 2 took action {action}")
            
            # Take action
            state, reward, done, _ = env.step(action)
            turn += 1
        
        # Determine winner
        if reward > 0:
            if env.current_player == 0:
                agent1_wins += 1
                if verbose:
                    print(f"Game {game+1}: Agent 1 wins")
            else:
                agent2_wins += 1
                if verbose:
                    print(f"Game {game+1}: Agent 2 wins")
        elif reward < 0:
            if env.current_player == 0:
                agent2_wins += 1
                if verbose:
                    print(f"Game {game+1}: Agent 2 wins")
            else:
                agent1_wins += 1
                if verbose:
                    print(f"Game {game+1}: Agent 1 wins")
        else:
            draws += 1
            if verbose:
                print(f"Game {game+1}: Draw")
    
    # Calculate win rates
    agent1_win_rate = agent1_wins / num_games
    agent2_win_rate = agent2_wins / num_games
    draw_rate = draws / num_games
    
    return {
        "agent1_wins": agent1_wins,
        "agent2_wins": agent2_wins,
        "draws": draws,
        "agent1_win_rate": agent1_win_rate,
        "agent2_win_rate": agent2_win_rate,
        "draw_rate": draw_rate
    }

def train_all_models(episodes=1000):
    """Train all models"""
    print("\n=== Training Models ===\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Train DQN
    print("\nTraining DQN...")
    train_dqn(episodes=episodes, save_path="models/improved_dqn_final.pt")
    
    # Train REINFORCE
    print("\nTraining REINFORCE...")
    train_reinforce(episodes=episodes, save_path="models/improved_reinforce_final.pt")
    
    # Train with self-play
    print("\nTraining with self-play...")
    self_play_training(episodes=episodes)

def print_ascii_plot(values, width=50, height=10, title=""):
    """Create an ASCII plot of the values."""
    if not values:
        return ""
    
    # Get min and max values
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        max_val = min_val + 1  # Prevent division by zero
    
    # Create the plot
    plot = [f"\n{title} {'=' * (width-len(title))}\n"]
    
    # Create y-axis labels and plot points
    for i in range(height-1, -1, -1):
        # Y-axis label
        y_val = min_val + (max_val - min_val) * i / (height-1)
        plot.append(f"{y_val:6.2f} |")
        
        # Plot points
        for j in range(width):
            if j < len(values):
                idx = j * len(values) // width  # Sample points evenly
                val = values[idx]
                normalized = (val - min_val) / (max_val - min_val)
                if normalized * (height-1) >= i:
                    plot.append("█")
                else:
                    plot.append(" ")
        plot.append("\n")
    
    # X-axis
    plot.append("       " + "-" * width + "\n")
    plot.append("       " + "0" + " " * (width-8) + f"{len(values)-1}\n")
    
    return "".join(plot)

def print_progress(metrics: Dict[str, List[float]], episode: int) -> None:
    """Print training progress with metrics."""
    # Get recent metrics
    window = min(100, len(metrics['episode_rewards']))
    recent_rewards = metrics['episode_rewards'][-window:]
    recent_policy_losses = metrics['policy_losses'][-window:]
    recent_value_losses = metrics['value_losses'][-window:]
    recent_entropy = metrics['entropies'][-window:]
    
    # Clear previous line and print current metrics
    print("\033[K", end="")  # Clear line
    print(f"\rEpisode {episode}")
    print(f"Recent Stats (last {window} episodes):")
    print(f"Reward:       {np.mean(recent_rewards):.3f} (±{np.std(recent_rewards):.3f})")
    print(f"Policy Loss:  {np.mean(recent_policy_losses):.3f} (±{np.std(recent_policy_losses):.3f})")
    print(f"Value Loss:   {np.mean(recent_value_losses):.3f} (±{np.std(recent_value_losses):.3f})")
    print(f"Entropy:      {np.mean(recent_entropy):.3f} (±{np.std(recent_entropy):.3f})")
    
    if metrics['eval_rewards']:
        print(f"Latest Eval:   {metrics['eval_rewards'][-1]:.3f}")
        print(f"Best Eval:     {max(metrics['eval_rewards']):.3f}")
    
    print(f"Best Reward:   {max(metrics['episode_rewards']):.3f}")
    print("-" * 50)

def train_reinforce(
    env: ImprovedGinRummyEnv,
    agent: REINFORCEAgent,
    num_episodes: int,
    eval_interval: int = 100,
    save_dir: str = "models"
) -> Dict[str, List[float]]:
    """Train REINFORCE agent and track metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = {
        'episode_rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'eval_rewards': []
    }
    
    # Create rules-based opponent for evaluation
    eval_opponent = RulesBasedAgent()
    
    # Disable environment printing
    env.print_state = lambda: None
    
    # Progress bar for episodes
    progress_bar = tqdm(range(num_episodes), desc="Training Progress")
    
    for episode in progress_bar:
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Episode progress bar
        episode_steps = tqdm(total=100, desc="Episode Steps", leave=False)
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience
            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            state = next_state
            episode_reward += reward
            episode_steps.update(1)
        
        episode_steps.close()
        
        # Train on episode
        losses = agent.train()
        
        # Record metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['policy_losses'].append(losses['policy_loss'])
        metrics['value_losses'].append(losses['value_loss'])
        metrics['entropies'].append(losses['entropy'])
        
        # Update progress bar with current metrics
        progress_bar.set_postfix({
            'reward': f"{episode_reward:.2f}",
            'policy_loss': f"{losses['policy_loss']:.2f}",
            'value_loss': f"{losses['value_loss']:.2f}",
            'entropy': f"{losses['entropy']:.2f}"
        })
        
        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_against_rules(agent, eval_opponent, env, num_games=50)
            metrics['eval_rewards'].append(eval_reward)
            
            # Save model
            agent.save(os.path.join(save_dir, f"reinforce_episode_{episode+1}.pt"))
            
            # Print progress
            print_progress(metrics, episode + 1)
    
    return metrics

def evaluate_against_rules(
    agent: REINFORCEAgent,
    opponent: RulesBasedAgent,
    env: ImprovedGinRummyEnv,
    num_games: int = 100
) -> float:
    """Evaluate agent against rules-based opponent."""
    total_reward = 0
    
    # Disable environment printing
    env.print_state = lambda: None
    
    # Progress bar for evaluation games
    for _ in tqdm(range(num_games), desc="Evaluating", leave=False):
        state = env.reset()
        done = False
        
        while not done:
            # Agent's turn
            if env.current_player == 0:
                action = agent.select_action(state)
            else:
                action = opponent.select_action(state)
            
            state, reward, done, truncated, info = env.step(action)
            if env.current_player == 0:  # Only count agent's rewards
                total_reward += reward
    
    return total_reward / num_games

def plot_metrics(metrics: Dict[str, List[float]], episode: int) -> None:
    """Plot training metrics in real-time."""
    plt.clf()  # Clear the current figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics (Episode {episode})')
    
    # Plot episode rewards
    ax1.plot(metrics['episode_rewards'], label='Episode Reward')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    
    # Plot losses
    ax2.plot(metrics['policy_losses'], label='Policy Loss', color='red')
    ax2.plot(metrics['value_losses'], label='Value Loss', color='blue')
    ax2.set_title('Training Losses')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    # Plot entropy
    ax3.plot(metrics['entropies'], label='Entropy', color='green')
    ax3.set_title('Policy Entropy')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Entropy')
    ax3.legend()
    
    # Plot evaluation rewards
    eval_episodes = list(range(0, episode + 1, 100))[1:]  # Every 100 episodes
    if metrics['eval_rewards']:  # Only plot if we have evaluation data
        ax4.plot(eval_episodes, metrics['eval_rewards'], label='Eval Reward', color='purple')
        ax4.set_title('Evaluation Rewards')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average Reward')
        ax4.legend()
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Save the current plot
    plt.savefig(f'plots/training_metrics_episode_{episode}.png')
    
    # Show the plot
    plt.show()
    plt.pause(0.1)  # Small pause to update the plots

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train and evaluate Gin Rummy agents')
    parser.add_argument('--train', action='store_true', help='Train the agents')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the agents')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes for training')
    parser.add_argument('--eval-games', type=int, default=1000, help='Number of games for evaluation')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    args = parser.parse_args()
    
    if not args.train and not args.evaluate:
        parser.print_help()
        return
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.train:
        # Create environments and agents
        env_reinforce = ImprovedGinRummyEnv()
        env_dqn = ImprovedGinRummyEnv()
        env_mcts = ImprovedGinRummyEnv()
        
        reinforce_agent = REINFORCEAgent()
        dqn_agent = ImprovedDQNAgent()
        mcts_agent = MCTSAgent()
        
        # Disable all printing from environments
        for env in [env_reinforce, env_dqn, env_mcts]:
            env.print_state = lambda: None
            env.print_action = lambda x: None
            env.print_reward = lambda x: None
        
        # Dictionary to store metrics for all agents
        all_metrics = {
            'reinforce': {'metrics': None, 'agent': reinforce_agent},
            'dqn': {'metrics': None, 'agent': dqn_agent},
            'mcts': {'metrics': None, 'agent': mcts_agent}
        }
        
        try:
            print("Starting parallel training of all agents...")
            print("Press Ctrl+C to stop training and save current progress")
            
            # Train REINFORCE
            metrics_reinforce = train_reinforce(
                env=env_reinforce,
                agent=reinforce_agent,
                num_episodes=args.episodes,
                eval_interval=100
            )
            all_metrics['reinforce']['metrics'] = metrics_reinforce
            
            # Train DQN
            metrics_dqn = train_dqn(
                env=env_dqn,
                agent=dqn_agent,
                num_episodes=args.episodes,
                eval_interval=100
            )
            all_metrics['dqn']['metrics'] = metrics_dqn
            
            # Train MCTS
            metrics_mcts = train_mcts(
                env=env_mcts,
                agent=mcts_agent,
                num_episodes=args.episodes,
                eval_interval=100
            )
            all_metrics['mcts']['metrics'] = metrics_mcts
            
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving current progress...")
            
            # Save all agents' current state
            for name, data in all_metrics.items():
                if data['metrics'] is not None:
                    save_path = os.path.join('models', f'{name}_interrupted.pt')
                    data['agent'].save(save_path)
                    print(f"Saved {name} model to {save_path}")
            
            print("All progress saved. You can resume training later.")
            return
        
        # Save final models
        for name, data in all_metrics.items():
            if data['metrics'] is not None:
                save_path = os.path.join('models', f'{name}_final.pt')
                data['agent'].save(save_path)
                print(f"Saved {name} model to {save_path}")
    
    if args.evaluate:
        # Create environment and agents
        env = ImprovedGinRummyEnv()
        reinforce_agent = REINFORCEAgent()
        dqn_agent = ImprovedDQNAgent()
        mcts_agent = MCTSAgent()
        rules_agent = RulesBasedAgent()
        
        # Disable environment printing
        env.print_state = lambda: None
        env.print_action = lambda x: None
        env.print_reward = lambda x: None
        
        # Load latest models if available
        for agent_name, agent in [
            ('reinforce', reinforce_agent),
            ('dqn', dqn_agent),
            ('mcts', mcts_agent)
        ]:
            model_files = [f for f in os.listdir('models') if f.startswith(f'{agent_name}_')]
            if model_files:
                latest_model = max(model_files, key=lambda x: int(x.split('_')[2].split('.')[0]) if x.split('_')[2].split('.')[0].isdigit() else 0)
                print(f"Loading {agent_name} model: {latest_model}")
                agent.load(os.path.join('models', latest_model))
        
        # Evaluate all agents against rules-based agent
        for agent_name, agent in [
            ('REINFORCE', reinforce_agent),
            ('DQN', dqn_agent),
            ('MCTS', mcts_agent)
        ]:
            final_reward = evaluate_against_rules(
                agent=agent,
                opponent=rules_agent,
                env=env,
                num_games=args.eval_games
            )
            print(f"{agent_name} evaluation reward against rules-based agent: {final_reward:.2f}")

if __name__ == "__main__":
    main() 