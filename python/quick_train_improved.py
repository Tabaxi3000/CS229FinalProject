#!/usr/bin/env python3

import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Import our models and environment
from dqn import DQNAgent, DQNetwork
from reinforce import REINFORCEAgent, PolicyNetwork
from mcts import MCTSAgent, PolicyValueNetwork
from gin_rummy_env import GinRummyEnv
from improved_training import ImprovedDQNAgent, ImprovedREINFORCEAgent

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Constants for quick training
QUICK_EPISODES = 50  # Increased from 20 to 50 for better training
EVAL_INTERVAL = 10
SAVE_INTERVAL = 20
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 5
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 10000
REWARD_SCALING = 10.0
LEARNING_RATE = 0.001
GAMMA = 0.99

class ReplayMemory:
    """Replay memory for DQN training"""
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    def __len__(self):
        return len(self.memory)

def quick_train_dqn():
    """Quick training for DQN agent"""
    print("Starting quick DQN training...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Initialize environment and agent
    env = GinRummyEnv()
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    
    policy_net = DQNetwork().to(device)
    target_net = DQNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    
    # Training metrics
    rewards = []
    win_rates = []
    
    # Training loop with progress bar
    print(f"Training DQN for {QUICK_EPISODES} episodes...")
    episode_pbar = tqdm(range(1, QUICK_EPISODES + 1), desc="DQN Training")
    
    steps_done = 0
    for episode in episode_pbar:
        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Play one episode
        while not done:
            # Select action with epsilon-greedy
            epsilon = max(EPSILON_END, EPSILON_START - (steps_done / EPSILON_DECAY))
            
            # Move state to device
            state_device = {
                'hand_matrix': state['hand_matrix'].to(device),
                'discard_history': state['discard_history'].to(device),
                'valid_actions_mask': state['valid_actions_mask'].to(device)
            }
            
            # Get valid actions
            valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_actions, int):
                valid_actions = [valid_actions]
                
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Random action
                action = random.choice(valid_actions)
            else:
                # Greedy action
                with torch.no_grad():
                    q_values = policy_net(
                        state_device['hand_matrix'],
                        state_device['discard_history'],
                        state_device['valid_actions_mask']
                    )
                    # Mask invalid actions
                    mask = state_device['valid_actions_mask']
                    if mask.dim() == 1 and q_values.dim() > 1:
                        mask = mask.unsqueeze(0)
                        if mask.size(1) != q_values.size(1):
                            # Skip masking if dimensions don't match
                            pass
                        else:
                            mask = mask.expand_as(q_values)
                            q_values = q_values.masked_fill(~mask.bool(), float('-inf'))
                    else:
                        q_values = q_values.masked_fill(~mask.bool(), float('-inf'))
                    action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Scale reward for better learning
            scaled_reward = reward * REWARD_SCALING
            
            # Store transition in memory
            memory.push(state, action, scaled_reward, next_state, done)
            
            # Move to next state
            state = next_state
            episode_reward += reward
            
            # Optimize model if enough samples
            if len(memory) >= BATCH_SIZE:
                # Sample batch
                transitions = memory.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                
                # Create batch tensors
                state_batch = {
                    'hand_matrix': torch.cat([s['hand_matrix'] for s in batch_state]).to(device),
                    'discard_history': torch.cat([s['discard_history'] for s in batch_state]).to(device),
                    'valid_actions_mask': torch.cat([s['valid_actions_mask'] for s in batch_state]).to(device)
                }
                
                next_state_batch = {
                    'hand_matrix': torch.cat([s['hand_matrix'] for s in batch_next_state]).to(device),
                    'discard_history': torch.cat([s['discard_history'] for s in batch_next_state]).to(device),
                    'valid_actions_mask': torch.cat([s['valid_actions_mask'] for s in batch_next_state]).to(device)
                }
                
                action_batch = torch.tensor(batch_action, device=device).unsqueeze(1)
                reward_batch = torch.tensor(batch_reward, device=device)
                done_batch = torch.tensor(batch_done, device=device, dtype=torch.float32)
                
                # Compute Q(s_t, a)
                try:
                    q_values = policy_net(
                        state_batch['hand_matrix'],
                        state_batch['discard_history'],
                        state_batch['valid_actions_mask']
                    )
                    state_action_values = q_values.gather(1, action_batch)
                except RuntimeError as e:
                    print(f"Error in batch processing: {e}")
                    # Skip this batch if there's an error
                    continue
                
                # Compute V(s_{t+1}) for all next states
                with torch.no_grad():
                    try:
                        next_q_values = target_net(
                            next_state_batch['hand_matrix'],
                            next_state_batch['discard_history'],
                            next_state_batch['valid_actions_mask']
                        )
                        next_state_values = next_q_values.max(1)[0]
                        
                        # Set V(s) = 0 for terminal states
                        next_state_values = next_state_values * (1 - done_batch)
                    except RuntimeError as e:
                        print(f"Error in next state processing: {e}")
                        # Skip this batch if there's an error
                        continue
                    
                    # Compute the expected Q values
                    expected_state_action_values = (next_state_values * 0.99) + reward_batch
                    
                # Compute Huber loss
                loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            
            # Update target network
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            # Increment step counter
            steps_done += 1
        
        # Track metrics
        rewards.append(episode_reward)
        
        # Evaluate agent periodically
        if episode % EVAL_INTERVAL == 0:
            win_rate = evaluate_dqn(policy_net, device, env, num_games=10)
            win_rates.append(win_rate)
            print(f"\nEpisode {episode}/{QUICK_EPISODES} - Win rate: {win_rate:.2f}")
        
        # Save model periodically
        if episode % SAVE_INTERVAL == 0:
            torch.save({
                'policy_net': policy_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'steps_done': steps_done
            }, f"models/quick_dqn_episode_{episode}.pt")
            
        # Update progress bar
        avg_reward = sum(rewards[-min(10, len(rewards)):]) / min(10, len(rewards))
        episode_pbar.set_postfix(reward=f"{avg_reward:.2f}", epsilon=f"{epsilon:.2f}")
    
    # Save final model
    torch.save({
        'policy_net': policy_net.state_dict(),
        'target_net': target_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'steps_done': steps_done
    }, "models/quick_dqn_final.pt")
    
    print("Quick DQN training complete!")
    return policy_net

def quick_train_reinforce():
    """Quick training for REINFORCE agent"""
    print("Starting quick REINFORCE training...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Initialize environment and agent
    env = GinRummyEnv()
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    
    policy = PolicyNetwork().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # Training metrics
    rewards = []
    win_rates = []
    
    # Training loop with progress bar
    print(f"Training REINFORCE for {QUICK_EPISODES} episodes...")
    episode_pbar = tqdm(range(1, QUICK_EPISODES + 1), desc="REINFORCE Training")
    
    for episode in episode_pbar:
        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Storage for episode
        log_probs = []
        episode_rewards = []
        entropies = []
        
        # Play one episode
        while not done:
            # Move state to device
            state_device = {
                'hand_matrix': state['hand_matrix'].to(device),
                'discard_history': state['discard_history'].to(device),
                'valid_actions_mask': state['valid_actions_mask'].to(device),
                'opponent_model': state.get('opponent_model', torch.zeros(52)).to(device)
            }
            
            # Forward pass through policy network
            action_probs, _ = policy(
                state_device['hand_matrix'],
                state_device['discard_history']
            )
            
            # Apply mask to action probabilities
            action_probs = action_probs.squeeze()
            action_probs = action_probs * state_device['valid_actions_mask']
            
            # Normalize probabilities
            action_probs = action_probs / (action_probs.sum() + 1e-8)
            
            # Sample action from the distribution
            m = torch.distributions.Categorical(action_probs)
            action = m.sample().item()
            
            # Store log probability and entropy for training
            log_probs.append(m.log_prob(torch.tensor([action], device=device)))
            entropies.append(m.entropy().unsqueeze(0))  # Ensure entropy is at least 1D
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Scale reward for better learning
            scaled_reward = reward * REWARD_SCALING
            
            # Store reward
            episode_rewards.append(scaled_reward)
            
            # Move to next state
            state = next_state
            episode_reward += reward
        
        # Calculate discounted returns
        R = 0
        returns = []
        for r in reversed(episode_rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, device=device)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        policy_loss = torch.cat(policy_loss).sum()
        
        # Add entropy regularization
        if entropies:
            entropy_loss = -0.01 * torch.cat(entropies).sum()
        else:
            entropy_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = policy_loss + entropy_loss
        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        rewards.append(episode_reward)
        
        # Evaluate agent periodically
        if episode % EVAL_INTERVAL == 0:
            win_rate = evaluate_reinforce(policy, device, env, num_games=10)
            win_rates.append(win_rate)
            print(f"\nEpisode {episode}/{QUICK_EPISODES} - Win rate: {win_rate:.2f}")
        
        # Save model periodically
        if episode % SAVE_INTERVAL == 0:
            torch.save({
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f"models/quick_reinforce_episode_{episode}.pt")
            
        # Update progress bar
        avg_reward = sum(rewards[-min(10, len(rewards)):]) / min(10, len(rewards))
        episode_pbar.set_postfix(reward=f"{avg_reward:.2f}", loss=f"{loss.item():.4f}")
    
    # Save final model
    torch.save({
        'policy': policy.state_dict(),
        'optimizer': optimizer.state_dict()
    }, "models/quick_reinforce_final.pt")
    
    print("Quick REINFORCE training complete!")
    return policy

def evaluate_dqn(policy_net, device, env, num_games=10):
    """Evaluate DQN agent against random opponent"""
    wins = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            # Agent's turn
            if env.current_player == 0:
                # Move state to device
                state_device = {
                    'hand_matrix': state['hand_matrix'].to(device),
                    'discard_history': state['discard_history'].to(device),
                    'valid_actions_mask': state['valid_actions_mask'].to(device)
                }
                
                # Get action from policy network
                with torch.no_grad():
                    q_values = policy_net(
                        state_device['hand_matrix'],
                        state_device['discard_history'],
                        state_device['valid_actions_mask']
                    )
                    # Mask invalid actions
                    mask = state_device['valid_actions_mask']
                    if mask.dim() == 1 and q_values.dim() > 1:
                        mask = mask.unsqueeze(0)
                        if mask.size(1) != q_values.size(1):
                            # Skip masking if dimensions don't match
                            pass
                        else:
                            mask = mask.expand_as(q_values)
                            q_values = q_values.masked_fill(~mask.bool(), float('-inf'))
                    else:
                        q_values = q_values.masked_fill(~mask.bool(), float('-inf'))
                    action = q_values.argmax().item()
            # Random opponent's turn
            else:
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = random.choice(valid_actions)
                
            # Take action
            state, reward, done, _ = env.step(action)
            
        # Check if agent won
        if reward > 0:
            wins += 1
    
    return wins / num_games

def evaluate_reinforce(policy, device, env, num_games=10):
    """Evaluate REINFORCE agent against random opponent"""
    wins = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            # Agent's turn
            if env.current_player == 0:
                # Move state to device
                state_device = {
                    'hand_matrix': state['hand_matrix'].to(device),
                    'discard_history': state['discard_history'].to(device),
                    'valid_actions_mask': state['valid_actions_mask'].to(device),
                    'opponent_model': state.get('opponent_model', torch.zeros(52)).to(device)
                }
                
                # Get action from policy network
                with torch.no_grad():
                    action_probs, _ = policy(
                        state_device['hand_matrix'],
                        state_device['discard_history']
                    )
                    
                    # Apply mask to action probabilities
                    action_probs = action_probs.squeeze()
                    action_probs = action_probs * state_device['valid_actions_mask']
                    
                    # Normalize probabilities
                    action_probs = action_probs / (action_probs.sum() + 1e-8)
                    
                    action = action_probs.argmax().item()
            # Random opponent's turn
            else:
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = random.choice(valid_actions)
                
            # Take action
            state, reward, done, _ = env.step(action)
            
        # Check if agent won
        if reward > 0:
            wins += 1
    
    return wins / num_games

def quick_train_mcts():
    """Quick training for MCTS agent"""
    print("Starting quick MCTS training...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Initialize environment and agent
    env = GinRummyEnv()
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    
    # Initialize policy-value network
    policy_value_net = PolicyValueNetwork().to(device)
    optimizer = torch.optim.Adam(policy_value_net.parameters(), lr=LEARNING_RATE)
    
    # Training metrics
    rewards = []
    win_rates = []
    policy_losses = []
    value_losses = []
    
    # Training loop with progress bar
    print(f"Training MCTS for {QUICK_EPISODES} episodes...")
    episode_pbar = tqdm(range(1, QUICK_EPISODES + 1), desc="MCTS Training")
    
    for episode in episode_pbar:
        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Storage for episode
        states = []
        policies = []
        values = []
        rewards_list = []
        
        # Play one episode
        while not done:
            # Move state to device
            state_device = {
                'hand_matrix': state['hand_matrix'].to(device),
                'discard_history': state['discard_history'].to(device),
                'valid_actions_mask': state['valid_actions_mask'].to(device),
                'opponent_model': state.get('opponent_model', torch.zeros(52)).to(device)
            }
            
            # Store state
            states.append(state_device)
            
            # Forward pass through policy-value network
            with torch.no_grad():
                policy_probs, state_value = policy_value_net(
                    state_device['hand_matrix'],
                    state_device['discard_history'],
                    state_device['opponent_model']
                )
                
                # Apply mask to policy probabilities
                policy_probs = policy_probs.squeeze()
                policy_probs = policy_probs * state_device['valid_actions_mask']
                
                # Normalize probabilities
                policy_probs = policy_probs / (policy_probs.sum() + 1e-8)
            
            # Store policy and value
            policies.append(policy_probs)
            values.append(state_value)
            
            # Sample action from the distribution
            valid_actions = torch.nonzero(state_device['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_actions, int):
                valid_actions = [valid_actions]
                
            # Use epsilon-greedy for exploration
            if random.random() < 0.1:  # 10% exploration
                action = random.choice(valid_actions)
            else:
                # Choose action with highest probability
                masked_probs = policy_probs.clone()
                masked_probs[~state_device['valid_actions_mask'].bool()] = float('-inf')
                action = masked_probs.argmax().item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store reward
            rewards_list.append(reward)
            
            # Move to next state
            state = next_state
            episode_reward += reward
        
        # Calculate returns with TD-lambda
        returns = []
        gae = 0
        next_value = 0
        for i in reversed(range(len(rewards_list))):
            delta = rewards_list[i] + GAMMA * next_value - values[i].item()
            gae = delta + GAMMA * 0.95 * gae
            next_value = values[i].item()
            returns.insert(0, gae + values[i].item())
        
        # Convert to tensors
        returns_tensor = torch.tensor(returns, device=device).unsqueeze(1)
        
        # Calculate policy and value losses
        policy_loss = 0
        value_loss = 0
        
        # Update in mini-batches
        indices = list(range(len(states)))
        random.shuffle(indices)
        
        for start_idx in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start_idx:start_idx + BATCH_SIZE]
            
            if not batch_indices:
                continue
                
            # Get batch data
            batch_states = [states[i] for i in batch_indices]
            batch_policies = [policies[i] for i in batch_indices]
            batch_returns = [returns_tensor[i] for i in batch_indices]
            
            # Forward pass
            state_batch = {
                'hand_matrix': torch.cat([s['hand_matrix'] for s in batch_states]),
                'discard_history': torch.cat([s['discard_history'] for s in batch_states]),
                'opponent_model': torch.cat([s['opponent_model'] for s in batch_states])
            }
            
            policy_probs, state_values = policy_value_net(
                state_batch['hand_matrix'],
                state_batch['discard_history'],
                state_batch['opponent_model']
            )
            
            # Calculate batch policy loss (cross-entropy)
            batch_policy_loss = 0
            for i, idx in enumerate(batch_indices):
                target_policy = policies[idx]
                log_probs = torch.log(policy_probs[i] + 1e-8)
                batch_policy_loss -= (target_policy * log_probs).sum()
            
            batch_policy_loss /= len(batch_indices)
            
            # Calculate batch value loss (MSE)
            batch_returns_tensor = torch.cat(batch_returns)
            batch_value_loss = F.mse_loss(state_values, batch_returns_tensor)
            
            # Total loss
            batch_loss = batch_policy_loss + batch_value_loss
            
            # Optimize
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_value_net.parameters(), 1.0)
            optimizer.step()
            
            # Track losses
            policy_loss += batch_policy_loss.item()
            value_loss += batch_value_loss.item()
        
        # Average losses
        if len(indices) > 0:
            policy_loss /= (len(indices) // BATCH_SIZE + 1)
            value_loss /= (len(indices) // BATCH_SIZE + 1)
        
        # Track metrics
        rewards.append(episode_reward)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        
        # Evaluate agent periodically
        if episode % EVAL_INTERVAL == 0:
            win_rate = evaluate_mcts(policy_value_net, device, env, num_games=5)
            win_rates.append(win_rate)
            print(f"\nEpisode {episode}/{QUICK_EPISODES} - Win rate: {win_rate:.2f}, Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}")
        
        # Save model periodically
        if episode % SAVE_INTERVAL == 0:
            torch.save(
                policy_value_net.state_dict(),
                f"models/quick_mcts_episode_{episode}.pt"
            )
        
        # Update progress bar
        episode_pbar.set_postfix(reward=f"{episode_reward:.2f}", p_loss=f"{policy_loss:.4f}", v_loss=f"{value_loss:.4f}")
    
    # Save final models
    torch.save(
        policy_value_net.state_dict(),
        "models/quick_mcts_policy_final.pt"
    )
    torch.save(
        policy_value_net.state_dict(),
        "models/quick_mcts_value_final.pt"
    )
    
    print("Quick MCTS training complete!")
    return policy_value_net

def evaluate_mcts(policy_value_net, device, env, num_games=5):
    """Evaluate MCTS agent against random opponent"""
    wins = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            # Agent's turn
            if env.current_player == 0:
                # Move state to device
                state_device = {
                    'hand_matrix': state['hand_matrix'].to(device),
                    'discard_history': state['discard_history'].to(device),
                    'valid_actions_mask': state['valid_actions_mask'].to(device),
                    'opponent_model': state.get('opponent_model', torch.zeros(52)).to(device)
                }
                
                # Get action from policy network
                with torch.no_grad():
                    policy_probs, _ = policy_value_net(
                        state_device['hand_matrix'],
                        state_device['discard_history'],
                        state_device['opponent_model']
                    )
                    
                    # Apply mask to policy probabilities
                    policy_probs = policy_probs.squeeze()
                    policy_probs = policy_probs * state_device['valid_actions_mask']
                    
                    # Normalize probabilities
                    policy_probs = policy_probs / (policy_probs.sum() + 1e-8)
                    
                    # Choose action with highest probability
                    action = policy_probs.argmax().item()
            # Random opponent's turn
            else:
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = random.choice(valid_actions)
                
            # Take action
            state, reward, done, _ = env.step(action)
            
        # Check if agent won
        if reward > 0:
            wins += 1
    
    return wins / num_games

def main():
    """Main function"""
    print("Starting quick training for Gin Rummy AI agents...")
    
    # Train DQN
    dqn_policy = quick_train_dqn()
    
    # Train REINFORCE
    reinforce_policy = quick_train_reinforce()
    
    # Train MCTS
    mcts_policy = quick_train_mcts()
    
    print("Quick training complete!")

if __name__ == "__main__":
    main() 