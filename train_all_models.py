#!/usr/bin/env python3

import os
import subprocess
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def ensure_directory_exists(directory):
    """Ensure that a directory exists, creating it if necessary."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def run_command(command, cwd=None):
    """Run a command and return its output."""
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, 
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                           text=True)
    if result.returncode != 0:
        print(f"Command failed with error: {result.stderr}")
    return result.stdout

def train_dqn(episodes=1000, eval_interval=100, save_interval=200):
    """Train the DQN model."""
    print("\n=== Training DQN Model ===")
    
    # Ensure the metrics directory exists
    ensure_directory_exists("metrics")
    
    # Run the training command
    command = f"python python/improved_dqn_train.py --episodes {episodes} --eval-interval {eval_interval} --save-interval {save_interval} --reward-shaping --deadwood-reward-scale 0.05 --win-reward 3.0 --gin-reward 4.0 --knock-reward 1.5 --save"
    output = run_command(command)
    
    # Extract metrics from the output
    rewards = []
    losses = []
    win_rates = []
    
    # Parse the output to extract metrics
    for line in output.split('\n'):
        if "Episode" in line and "reward:" in line:
            try:
                reward = float(line.split("reward:")[1].split(",")[0].strip())
                rewards.append(reward)
            except:
                pass
        
        if "loss:" in line:
            try:
                loss = float(line.split("loss:")[1].split(",")[0].strip())
                losses.append(loss)
            except:
                pass
        
        if "Win rate:" in line:
            try:
                win_rate = float(line.split("Win rate:")[1].split()[0].strip())
                win_rates.append(win_rate)
            except:
                pass
    
    # Save metrics to file
    metrics = {
        "rewards": rewards,
        "losses": losses,
        "win_rates": win_rates,
        "eval_interval": eval_interval
    }
    
    with open("metrics/dqn_metrics.json", "w") as f:
        json.dump(metrics, f)
    
    return metrics

def train_reinforce(episodes=1000, eval_interval=100, save_interval=200):
    """Train the REINFORCE model."""
    print("\n=== Training REINFORCE Model ===")
    
    # Ensure the metrics directory exists
    ensure_directory_exists("metrics")
    
    # Run the training command
    command = f"python python/train_reinforce.py --episodes {episodes} --eval-interval {eval_interval} --save-interval {save_interval} --reward-shaping --deadwood-reward-scale 0.05 --win-reward 3.0 --gin-reward 4.0 --knock-reward 1.5 --save"
    output = run_command(command)
    
    # Extract metrics from the output
    rewards = []
    losses = []
    win_rates = []
    
    # Parse the output to extract metrics
    for line in output.split('\n'):
        if "Episode" in line and "reward:" in line:
            try:
                reward = float(line.split("reward:")[1].split(",")[0].strip())
                rewards.append(reward)
            except:
                pass
        
        if "loss:" in line:
            try:
                loss = float(line.split("loss:")[1].split(",")[0].strip())
                losses.append(loss)
            except:
                pass
        
        if "Win rate:" in line:
            try:
                win_rate = float(line.split("Win rate:")[1].split()[0].strip())
                win_rates.append(win_rate)
            except:
                pass
    
    # Save metrics to file
    metrics = {
        "rewards": rewards,
        "losses": losses,
        "win_rates": win_rates,
        "eval_interval": eval_interval
    }
    
    with open("metrics/reinforce_metrics.json", "w") as f:
        json.dump(metrics, f)
    
    return metrics

def train_mcts(episodes=1000, eval_interval=100, save_interval=200):
    """Train the MCTS model."""
    print("\n=== Training MCTS Model ===")
    
    # Ensure the metrics directory exists
    ensure_directory_exists("metrics")
    
    # Run the training command
    command = f"python python/train_mcts.py --episodes {episodes} --eval-interval {eval_interval} --save-interval {save_interval} --reward-shaping --deadwood-reward-scale 0.05 --win-reward 3.0 --gin-reward 4.0 --knock-reward 1.5 --simulations 50 --save"
    output = run_command(command)
    
    # Extract metrics from the output
    rewards = []
    policy_losses = []
    value_losses = []
    win_rates = []
    
    # Parse the output to extract metrics
    for line in output.split('\n'):
        if "reward:" in line:
            try:
                reward = float(line.split("reward:")[1].split(",")[0].strip())
                rewards.append(reward)
            except:
                pass
        
        if "p_loss:" in line:
            try:
                p_loss = float(line.split("p_loss:")[1].split(",")[0].strip())
                policy_losses.append(p_loss)
            except:
                pass
        
        if "v_loss:" in line:
            try:
                v_loss = float(line.split("v_loss:")[1].split()[0].strip())
                value_losses.append(v_loss)
            except:
                pass
        
        if "win_rate:" in line:
            try:
                win_rate = float(line.split("win_rate:")[1].split(",")[0].strip())
                win_rates.append(win_rate)
            except:
                pass
    
    # Save metrics to file
    metrics = {
        "rewards": rewards,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "win_rates": win_rates,
        "eval_interval": eval_interval
    }
    
    with open("metrics/mcts_metrics.json", "w") as f:
        json.dump(metrics, f)
    
    return metrics

def plot_learning_curves(dqn_metrics=None, reinforce_metrics=None, mcts_metrics=None):
    """Plot learning curves for all trained models."""
    print("\n=== Generating Learning Curves ===")
    
    # Ensure the plots directory exists
    ensure_directory_exists("plots")
    
    # Load metrics from files if not provided
    if dqn_metrics is None and os.path.exists("metrics/dqn_metrics.json"):
        with open("metrics/dqn_metrics.json", "r") as f:
            dqn_metrics = json.load(f)
    
    if reinforce_metrics is None and os.path.exists("metrics/reinforce_metrics.json"):
        with open("metrics/reinforce_metrics.json", "r") as f:
            reinforce_metrics = json.load(f)
    
    if mcts_metrics is None and os.path.exists("metrics/mcts_metrics.json"):
        with open("metrics/mcts_metrics.json", "r") as f:
            mcts_metrics = json.load(f)
    
    # Plot rewards
    plt.figure(figsize=(12, 8))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    if dqn_metrics:
        plt.plot(dqn_metrics["rewards"], label="DQN")
    if reinforce_metrics:
        plt.plot(reinforce_metrics["rewards"], label="REINFORCE")
    if mcts_metrics:
        plt.plot(mcts_metrics["rewards"], label="MCTS")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    # Plot losses
    plt.subplot(2, 2, 2)
    if dqn_metrics and "losses" in dqn_metrics:
        plt.plot(dqn_metrics["losses"], label="DQN")
    if reinforce_metrics and "losses" in reinforce_metrics:
        plt.plot(reinforce_metrics["losses"], label="REINFORCE")
    if mcts_metrics and "policy_losses" in mcts_metrics:
        plt.plot(mcts_metrics["policy_losses"], label="MCTS (Policy)")
    plt.title("Training Losses")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot win rates
    plt.subplot(2, 2, 3)
    if dqn_metrics and "win_rates" in dqn_metrics:
        eval_interval = dqn_metrics.get("eval_interval", 100)
        x_dqn = np.arange(0, len(dqn_metrics["win_rates"]) * eval_interval, eval_interval)
        plt.plot(x_dqn, dqn_metrics["win_rates"], label="DQN", marker='o')
    
    if reinforce_metrics and "win_rates" in reinforce_metrics:
        eval_interval = reinforce_metrics.get("eval_interval", 100)
        x_reinforce = np.arange(0, len(reinforce_metrics["win_rates"]) * eval_interval, eval_interval)
        plt.plot(x_reinforce, reinforce_metrics["win_rates"], label="REINFORCE", marker='s')
    
    if mcts_metrics and "win_rates" in mcts_metrics:
        eval_interval = mcts_metrics.get("eval_interval", 100)
        x_mcts = np.arange(0, len(mcts_metrics["win_rates"]) * eval_interval, eval_interval)
        plt.plot(x_mcts, mcts_metrics["win_rates"], label="MCTS", marker='^')
    
    plt.title("Win Rates")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.legend()
    
    # Plot MCTS specific metrics if available
    plt.subplot(2, 2, 4)
    if mcts_metrics and "value_losses" in mcts_metrics:
        plt.plot(mcts_metrics["value_losses"], label="Value Loss")
        plt.title("MCTS Value Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("plots/learning_curves.png")
    print("Learning curves saved to plots/learning_curves.png")
    
    # Create a comparison plot of win rates
    plt.figure(figsize=(10, 6))
    
    if dqn_metrics and "win_rates" in dqn_metrics:
        eval_interval = dqn_metrics.get("eval_interval", 100)
        x_dqn = np.arange(0, len(dqn_metrics["win_rates"]) * eval_interval, eval_interval)
        plt.plot(x_dqn, dqn_metrics["win_rates"], label="DQN", marker='o')
    
    if reinforce_metrics and "win_rates" in reinforce_metrics:
        eval_interval = reinforce_metrics.get("eval_interval", 100)
        x_reinforce = np.arange(0, len(reinforce_metrics["win_rates"]) * eval_interval, eval_interval)
        plt.plot(x_reinforce, reinforce_metrics["win_rates"], label="REINFORCE", marker='s')
    
    if mcts_metrics and "win_rates" in mcts_metrics:
        eval_interval = mcts_metrics.get("eval_interval", 100)
        x_mcts = np.arange(0, len(mcts_metrics["win_rates"]) * eval_interval, eval_interval)
        plt.plot(x_mcts, mcts_metrics["win_rates"], label="MCTS", marker='^')
    
    plt.title("Win Rate Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("plots/win_rate_comparison.png")
    print("Win rate comparison saved to plots/win_rate_comparison.png")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Gin Rummy RL models")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to train each model")
    parser.add_argument("--eval-interval", type=int, default=250, help="Interval for evaluation during training")
    parser.add_argument("--save-interval", type=int, default=500, help="Interval for saving models during training")
    parser.add_argument("--models", type=str, default="all", help="Models to train: 'dqn', 'reinforce', 'mcts', or 'all'")
    parser.add_argument("--plot-only", action="store_true", help="Only plot existing metrics without training")
    
    args = parser.parse_args()
    
    dqn_metrics = None
    reinforce_metrics = None
    mcts_metrics = None
    
    if not args.plot_only:
        # Train the selected models
        if args.models.lower() == "all" or "dqn" in args.models.lower():
            dqn_metrics = train_dqn(args.episodes, args.eval_interval, args.save_interval)
        
        if args.models.lower() == "all" or "reinforce" in args.models.lower():
            reinforce_metrics = train_reinforce(args.episodes, args.eval_interval, args.save_interval)
        
        if args.models.lower() == "all" or "mcts" in args.models.lower():
            mcts_metrics = train_mcts(args.episodes, args.eval_interval, args.save_interval)
    
    # Plot learning curves
    plot_learning_curves(dqn_metrics, reinforce_metrics, mcts_metrics)
    
    print("\n=== Training and Evaluation Complete ===")
    print("Models saved in the 'models' directory")
    print("Metrics saved in the 'metrics' directory")
    print("Plots saved in the 'plots' directory")

if __name__ == "__main__":
    main() 