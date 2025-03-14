#!/usr/bin/env python3

"""
CS229 Graph Generation Script

This script generates visualizations for the CS229 milestone based on
training and evaluation metrics.

Usage:
    python generate_cs229_graphs.py [--metrics-file METRICS_FILE] [--save-dir SAVE_DIR]

Author: CS229 Project Team
Date: March 2025
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate visualizations for CS229 milestone')
    parser.add_argument('--metrics-file', type=str, default='cs229_training_metrics.json',
                        help='Path to metrics file (default: cs229_training_metrics.json)')
    parser.add_argument('--save-dir', type=str, default='.',
                        help='Directory to save visualizations (default: current directory)')
    return parser.parse_args()

def load_metrics(metrics_file):
    """
    Load metrics from a JSON file.
    
    Args:
        metrics_file (str): Path to metrics file
        
    Returns:
        dict: Dictionary containing metrics
    """
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        print(f"Error: Metrics file '{metrics_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Metrics file '{metrics_file}' is not valid JSON.")
        sys.exit(1)

def plot_win_rate_comparison(metrics, save_dir):
    """
    Plot win rate comparison between different agents.
    
    Args:
        metrics (dict): Dictionary containing metrics
        save_dir (str): Directory to save visualization
    """
    plt.figure(figsize=(10, 6))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Define colors for each agent
    colors = {
        'DQN': '#1f77b4',  # Blue
        'REINFORCE': '#ff7f0e',  # Orange
        'MCTS': '#2ca02c',  # Green
        'IMPROVED_MCTS': '#d62728'  # Red
    }
    
    # Plot win rates for each agent
    episodes = metrics['episodes']
    for agent_type, win_rates in metrics['win_rates'].items():
        # Apply smoothing for better visualization
        window_size = max(1, len(win_rates) // 50)
        smoothed_win_rates = np.convolve(win_rates, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        
        plt.plot(smoothed_episodes, smoothed_win_rates, label=agent_type, color=colors[agent_type], linewidth=2)
    
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.title('Win Rate Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'cs229_win_rate_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved win rate comparison to {save_path}")
    
    # Save data for future reference
    data = {
        'episodes': episodes,
        'win_rates': metrics['win_rates']
    }
    data_path = os.path.join(save_dir, 'cs229_win_rates.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved win rate data to {data_path}")

def plot_learning_curves(metrics, save_dir):
    """
    Plot learning curves for different agents.
    
    Args:
        metrics (dict): Dictionary containing metrics
        save_dir (str): Directory to save visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Define colors for each agent
    colors = {
        'DQN': '#1f77b4',  # Blue
        'REINFORCE': '#ff7f0e',  # Orange
        'MCTS': '#2ca02c',  # Green
        'IMPROVED_MCTS': '#d62728'  # Red
    }
    
    # Plot win rates
    ax = axs[0, 0]
    episodes = metrics['episodes']
    for agent_type, win_rates in metrics['win_rates'].items():
        # Apply smoothing for better visualization
        window_size = max(1, len(win_rates) // 50)
        smoothed_win_rates = np.convolve(win_rates, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        
        ax.plot(smoothed_episodes, smoothed_win_rates, label=agent_type, color=colors[agent_type], linewidth=2)
    
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title('Win Rate Learning Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot rewards
    ax = axs[0, 1]
    for agent_type, rewards in metrics['rewards'].items():
        # Apply smoothing for better visualization
        window_size = max(1, len(rewards) // 50)
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        
        ax.plot(smoothed_episodes, smoothed_rewards, label=agent_type, color=colors[agent_type], linewidth=2)
    
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Reward Learning Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot game lengths
    ax = axs[1, 0]
    for agent_type, game_lengths in metrics['game_lengths'].items():
        # Apply smoothing for better visualization
        window_size = max(1, len(game_lengths) // 50)
        smoothed_game_lengths = np.convolve(game_lengths, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        
        ax.plot(smoothed_episodes, smoothed_game_lengths, label=agent_type, color=colors[agent_type], linewidth=2)
    
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Average Game Length', fontsize=12)
    ax.set_title('Game Length Learning Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot deadwood
    ax = axs[1, 1]
    for agent_type, deadwood in metrics['deadwood'].items():
        # Apply smoothing for better visualization
        window_size = max(1, len(deadwood) // 50)
        smoothed_deadwood = np.convolve(deadwood, np.ones(window_size)/window_size, mode='valid')
        smoothed_episodes = episodes[window_size-1:]
        
        ax.plot(smoothed_episodes, smoothed_deadwood, label=agent_type, color=colors[agent_type], linewidth=2)
    
    ax.set_xlabel('Episodes', fontsize=12)
    ax.set_ylabel('Average Deadwood', fontsize=12)
    ax.set_title('Deadwood Learning Curves', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'cs229_learning_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {save_path}")
    
    # Save data for future reference
    data = {
        'episodes': episodes,
        'win_rates': metrics['win_rates'],
        'rewards': metrics['rewards'],
        'game_lengths': metrics['game_lengths'],
        'deadwood': metrics['deadwood']
    }
    data_path = os.path.join(save_dir, 'cs229_learning_curves.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved learning curves data to {data_path}")

def plot_individual_metrics(metrics, save_dir):
    """
    Plot individual metrics for different agents.
    
    Args:
        metrics (dict): Dictionary containing metrics
        save_dir (str): Directory to save visualizations
    """
    # Define colors for each agent
    colors = {
        'DQN': '#1f77b4',  # Blue
        'REINFORCE': '#ff7f0e',  # Orange
        'MCTS': '#2ca02c',  # Green
        'IMPROVED_MCTS': '#d62728'  # Red
    }
    
    # Plot game length comparison
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Get final game lengths for each agent
    final_game_lengths = {agent_type: game_lengths[-1] for agent_type, game_lengths in metrics['game_lengths'].items()}
    
    # Create bar plot
    plt.bar(range(len(final_game_lengths)), list(final_game_lengths.values()), color=[colors[agent] for agent in final_game_lengths.keys()])
    plt.xticks(range(len(final_game_lengths)), list(final_game_lengths.keys()))
    plt.xlabel('Agent', fontsize=12)
    plt.ylabel('Average Game Length', fontsize=12)
    plt.title('Game Length Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'cs229_game_length.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved game length comparison to {save_path}")
    
    # Save data for future reference
    data = {
        'game_lengths': final_game_lengths
    }
    data_path = os.path.join(save_dir, 'cs229_game_lengths.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved game length data to {data_path}")
    
    # Plot deadwood comparison
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Get final deadwood for each agent
    final_deadwood = {agent_type: deadwood[-1] for agent_type, deadwood in metrics['deadwood'].items()}
    
    # Create bar plot
    plt.bar(range(len(final_deadwood)), list(final_deadwood.values()), color=[colors[agent] for agent in final_deadwood.keys()])
    plt.xticks(range(len(final_deadwood)), list(final_deadwood.keys()))
    plt.xlabel('Agent', fontsize=12)
    plt.ylabel('Average Deadwood', fontsize=12)
    plt.title('Deadwood Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'cs229_deadwood_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved deadwood comparison to {save_path}")
    
    # Save data for future reference
    data = {
        'deadwood': final_deadwood
    }
    data_path = os.path.join(save_dir, 'cs229_deadwood.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved deadwood data to {data_path}")
    
    # Plot reward comparison
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Get final rewards for each agent
    final_rewards = {agent_type: rewards[-1] for agent_type, rewards in metrics['rewards'].items()}
    
    # Create bar plot
    plt.bar(range(len(final_rewards)), list(final_rewards.values()), color=[colors[agent] for agent in final_rewards.keys()])
    plt.xticks(range(len(final_rewards)), list(final_rewards.keys()))
    plt.xlabel('Agent', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Reward Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'cs229_reward_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved reward comparison to {save_path}")
    
    # Save data for future reference
    data = {
        'rewards': final_rewards
    }
    data_path = os.path.join(save_dir, 'cs229_rewards.json')
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved reward data to {data_path}")

def main():
    """Main function to generate visualizations for CS229 milestone."""
    args = parse_arguments()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load metrics
    metrics = load_metrics(args.metrics_file)
    
    # Generate visualizations
    plot_win_rate_comparison(metrics, args.save_dir)
    plot_learning_curves(metrics, args.save_dir)
    plot_individual_metrics(metrics, args.save_dir)
    
    print("\nVisualization generation completed!")
    print(f"All visualizations saved to {args.save_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 