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
import json
from multiprocessing import Process, Event
from dqn import DQNAgent, DQNetwork
from reinforce import REINFORCEAgent, PolicyNetwork
from mcts import MCTSAgent, PolicyValueNetwork
from gin_rummy_env import GinRummyEnv
from improved_training import ImprovedDQNAgent, ImprovedREINFORCEAgent, train_dqn, train_reinforce, self_play_training
from rules_based_agent import RulesBasedAgent
from improved_gin_rummy_env import ImprovedGinRummyEnv
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
def evaluate_models(num_games=50, verbose=False):
    """Evaluate all models against random and against each other"""
    print("\n=== Evaluating Models ===\n")
    env = GinRummyEnv()
    dqn_agent = ImprovedDQNAgent()
    reinforce_agent = ImprovedREINFORCEAgent()
    dqn_agent.load("models/improved_dqn_final.pt")
    reinforce_agent.load("models/improved_reinforce_final.pt")
    class RandomAgent:
        def select_action(self, state, eval_mode=False):
            valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_actions, int):
                valid_actions = [valid_actions]
            return random.choice(valid_actions)
    random_agent = RandomAgent()
    print("\nEvaluating DQN vs Random...")
    dqn_vs_random_results = evaluate_agent_vs_agent(dqn_agent, random_agent, env, num_games, verbose)
    print("\nEvaluating REINFORCE vs Random...")
    reinforce_vs_random_results = evaluate_agent_vs_agent(reinforce_agent, random_agent, env, num_games, verbose)
    print("\nEvaluating DQN vs REINFORCE...")
    dqn_vs_reinforce_results = evaluate_agent_vs_agent(dqn_agent, reinforce_agent, env, num_games, verbose)
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
            if env.current_player == 0:
                action = agent1.select_action(state, eval_mode=True)
                if verbose:
                    print(f"Agent 1 took action {action}")
            else:
                action = agent2.select_action(state, eval_mode=True)
                if verbose:
                    print(f"Agent 2 took action {action}")
            state, reward, done, _ = env.step(action)
            turn += 1
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
    os.makedirs("models", exist_ok=True)
    print("\nTraining DQN...")
    train_dqn(episodes=episodes, save_path="models/improved_dqn_final.pt")
    print("\nTraining REINFORCE...")
    train_reinforce(episodes=episodes, save_path="models/improved_reinforce_final.pt")
    print("\nTraining with self-play...")
    self_play_training(episodes=episodes)
def print_ascii_plot(values, width=50, height=10, title=""):
    """Create an ASCII plot of the values."""
    if not values:
        return ""
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        max_val = min_val + 1  # Prevent division by zero
    plot = [f"\n{title} {'=' * (width-len(title))}\n"]
    for i in range(height-1, -1, -1):
        y_val = min_val + (max_val - min_val) * i / (height-1)
        plot.append(f"{y_val:6.2f} |")
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
    plot.append("       " + "-" * width + "\n")
    plot.append("       " + "0" + " " * (width-8) + f"{len(values)-1}\n")
    return "".join(plot)
def print_progress(metrics: Dict[str, List[float]], episode: int) -> None:
    """Print training progress with metrics."""
    window = min(100, len(metrics['episode_rewards']))
    recent_rewards = metrics['episode_rewards'][-window:]
    recent_policy_losses = metrics['policy_losses'][-window:]
    recent_value_losses = metrics['value_losses'][-window:]
    recent_entropy = metrics['entropies'][-window:]
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
def train_reinforce(env, agent, num_episodes=10000, eval_interval=100):
    """Train a REINFORCE agent."""
    pbar = tqdm(range(num_episodes), desc="Training REINFORCE")
    episode_rewards = []
    policy_losses = []
    value_losses = []
    win_rates = []
    rules_opponent = RulesBasedAgent()
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            state = next_state
            episode_reward += reward
        losses = agent.train()
        episode_rewards.append(episode_reward)
        if losses:
            policy_losses.append(losses['policy_loss'])
            value_losses.append(losses['value_loss'])
        if (episode + 1) % eval_interval == 0:
            win_rate = evaluate_agent(env, agent, num_games=20)
            win_rates.append(win_rate)
            avg_reward = sum(episode_rewards[-eval_interval:]) / eval_interval
            avg_policy_loss = sum(policy_losses[-eval_interval:]) / max(1, len(policy_losses[-eval_interval:]))
            avg_value_loss = sum(value_losses[-eval_interval:]) / max(1, len(value_losses[-eval_interval:]))
            pbar.set_postfix({
                'reward': f'{avg_reward:.2f}',
                'policy_loss': f'{avg_policy_loss:.2f}',
                'value_loss': f'{avg_value_loss:.2f}',
                'win_rate': f'{win_rate:.2f}'
            })
            metrics = {
                'episode_rewards': episode_rewards,
                'policy_losses': policy_losses,
                'value_losses': value_losses,
                'win_rates': win_rates,
                'eval_episodes': list(range(eval_interval, episode + 2, eval_interval))
            }
            with open('metrics/reinforce_metrics.json', 'w') as f:
                json.dump(metrics, f)
            plot_metrics(metrics, 'reinforce')
            if win_rate >= max(win_rates):
                agent.save('models/reinforce_best.pt')
                print(f"\nNew best model saved with win rate: {win_rate:.2f}")
    return episode_rewards, policy_losses, value_losses, win_rates
def save_metrics(metrics: Dict[str, Any], agent_type: str):
    """Save metrics to JSON file with proper error handling."""
    try:
        os.makedirs('metrics', exist_ok=True)
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, list)):
                serializable_metrics[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
            elif isinstance(value, (np.floating, np.integer)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        metrics_path = f'metrics/{agent_type}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        print(f"\nSaved metrics to {metrics_path}")
    except Exception as e:
        print(f"\nWarning: Failed to save metrics: {str(e)}")
def train_dqn(env, agent, num_episodes=10000, eval_interval=100):
    """Train a DQN agent."""
    pbar = tqdm(range(num_episodes), desc="Training DQN")
    episode_rewards = []
    losses = []
    win_rates = []
    epsilons = []
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        done = False
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.memory.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            loss = agent.optimize_model()
            if loss is not None:
                losses.append(loss)
            state = next_state
            episode_reward += reward
        episode_rewards.append(episode_reward)
        epsilons.append(epsilon)
        if (episode + 1) % eval_interval == 0:
            win_rate = evaluate_agent(env, agent, num_games=20)
            win_rates.append(win_rate)
            metrics = {
                'episode_rewards': episode_rewards,
                'losses': losses,
                'win_rates': win_rates,
                'epsilons': epsilons,
                'eval_episodes': list(range(eval_interval, episode + 2, eval_interval))
            }
            save_metrics(metrics, 'dqn')
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.subplot(1, 3, 2)
            if losses:
                plt.plot(losses)
                plt.title('Training Loss')
                plt.xlabel('Optimization Step')
                plt.ylabel('Loss')
            plt.subplot(1, 3, 3)
            plt.plot(range(eval_interval, episode + 2, eval_interval), win_rates)
            plt.title('Win Rate vs Random Opponent')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            plt.tight_layout()
            plt.savefig(f"plots/training_metrics_episode_{episode + 1}.png")
            plt.close()
            pbar.set_postfix({
                'reward': f'{episode_rewards[-1]:.2f}',
                'loss': f'{losses[-1]:.4f}' if losses else 'N/A',
                'epsilon': f'{epsilon:.2f}',
                'win_rate': f'{win_rate:.2f}'
            })
            if win_rate >= max(win_rates):
                agent.save('models/dqn_best.pt')
                print(f"\nNew best model saved with win rate: {win_rate:.2f}")
    return episode_rewards, losses, win_rates, epsilons
def train_mcts(env, agent, num_episodes=10000, eval_interval=100, save_path="models/mcts"):
    """Train an MCTS agent."""
    pbar = tqdm(range(num_episodes), desc="Training MCTS")
    episode_rewards = []
    win_rates = []
    best_win_rate = 0.0
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        done = False
        temperature = max(0.1, 1.0 - episode / num_episodes)
        while not done:
            action = agent.select_action(state, temperature=temperature)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
        episode_rewards.append(episode_reward)
        if (episode + 1) % eval_interval == 0:
            win_rate = evaluate_agent(env, agent, num_games=20)
            win_rates.append(win_rate)
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.save(f"{save_path}_best")
            metrics = {
                'episode_rewards': episode_rewards,
                'win_rates': win_rates,
                'eval_episodes': list(range(eval_interval, episode + 2, eval_interval))
            }
            with open(f"metrics/mcts_metrics.json", 'w') as f:
                json.dump(metrics, f)
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.subplot(1, 2, 2)
            plt.plot(range(eval_interval, episode + 2, eval_interval), win_rates)
            plt.title('Win Rate vs Random Opponent')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            plt.tight_layout()
            plt.savefig(f"plots/training_metrics_episode_{episode + 1}.png")
            plt.close()
            pbar.set_postfix({
                'Reward': episode_rewards[-1],
                'Win Rate': win_rates[-1],
                'Best Win Rate': best_win_rate
            })
    agent.save(f"{save_path}_final")
    return episode_rewards, win_rates
def evaluate_agent(env, agent, num_games=20):
    """Evaluate agent against random opponent."""
    wins = 0
    for _ in range(num_games):
        state = env.reset()
        done = False
        while not done:
            if env.current_player == 0:
                action = agent.select_action(state, temperature=0)
            else:
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = random.choice(valid_actions)
            state, reward, done, info = env.step(action)
        if info.get('winner', -1) == 0:
            wins += 1
    return wins / num_games
def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train and evaluate Gin Rummy agents')
    parser.add_argument('--train', action='store_true', help='Train the agents')
    parser.add_argument('--eval', action='store_true', help='Evaluate the agents')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes for training')
    parser.add_argument('--eval-games', type=int, default=100, help='Number of games for evaluation')
    args = parser.parse_args()
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    if args.train:
        print("Starting parallel training of all agents...")
        print("Press Ctrl+C to stop training and save current progress")
        env = ImprovedGinRummyEnv(verbose=False)  # Disable printing to avoid clutter
        dqn_agent = DQNAgent()
        mcts_agent = MCTSAgent()
        stop_event = Event()
        def train_dqn_process():
            try:
                train_dqn(env, dqn_agent, num_episodes=args.episodes)
            except KeyboardInterrupt:
                print("\nSaving DQN agent...")
                dqn_agent.save('models/dqn_interrupted.pt')
        def train_mcts_process():
            try:
                train_mcts(env, mcts_agent, num_episodes=args.episodes)
            except KeyboardInterrupt:
                print("\nSaving MCTS agent...")
                mcts_agent.save('models/mcts_interrupted.pt')
        dqn_process = Process(target=train_dqn_process)
        mcts_process = Process(target=train_mcts_process)
        try:
            dqn_process.start()
            mcts_process.start()
            dqn_process.join()
            mcts_process.join()
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving current state...")
            stop_event.set()
            dqn_process.join(timeout=5)
            mcts_process.join(timeout=5)
            if dqn_process.is_alive():
                dqn_process.terminate()
            if mcts_process.is_alive():
                mcts_process.terminate()
            print("Training stopped.")
    if args.eval:
        print("\nEvaluating agents...")
        env = ImprovedGinRummyEnv()
        dqn_agent = DQNAgent()
        mcts_agent = MCTSAgent()
        rules_agent = RulesBasedAgent()
        try:
            dqn_agent.load('models/dqn_best.pt')
            mcts_agent.load('models/mcts_best.pt')
        except:
            print("Warning: Could not load all model files. Using latest available models.")
        print("\nEvaluating against rules-based opponent...")
        dqn_wr = evaluate_agent(env, dqn_agent, num_games=args.eval_games)
        mcts_wr = evaluate_agent(env, mcts_agent, num_games=args.eval_games)
        print(f"\nResults against rules-based opponent ({args.eval_games} games):")
        print(f"DQN win rate: {dqn_wr:.2f}")
        print(f"MCTS win rate: {mcts_wr:.2f}")
if __name__ == '__main__':
    main() 
