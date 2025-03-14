import torch
from dqn import DQNAgent
from data_loader import GinRummyDataset
from tqdm import tqdm
import os

def train_dqn(num_epochs=100, batch_size=128, save_interval=10):
    # Initialize agent and dataset
    agent = DQNAgent()
    
    # Load all consolidated training data files
    datasets = []
    
    # Use the consolidated files instead of the original files
    for i in range(1, 11):  # Files are numbered 1 to 10
        file_path = f"../java/MavenProject/training_data_consolidated_{i}.json"
        if os.path.exists(file_path):
            datasets.append(GinRummyDataset(file_path))
    
    if not datasets:
        raise ValueError("No consolidated training data files found! Run consolidate_games.py first.")
    
    print(f"Loaded {len(datasets)} consolidated training data files")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train on each dataset
        for dataset_idx, dataset in enumerate(datasets):
            print(f"\nTraining on dataset {dataset_idx + 1}/{len(datasets)}")
            
            # Get dataset stats
            stats = dataset.get_stats()
            print(f"Dataset stats: {stats}")
            
            # Training iterations
            num_iterations = stats['total_states'] // batch_size
            progress_bar = tqdm(range(num_iterations), desc="Training")
            
            for _ in progress_bar:
                # Get batch of training data
                state_batch, action_batch, reward_batch, done_batch = dataset.get_training_data(batch_size)
                
                # Add experiences to replay buffer
                for i in range(batch_size):
                    agent.memory.push(
                        state={
                            'hand_matrix': state_batch['hand_matrix'][i:i+1],
                            'discard_history': state_batch['discard_history'][i:i+1],
                            'valid_actions_mask': state_batch['valid_actions_mask'][i:i+1]
                        },
                        action=torch.argmax(action_batch[i]).item(),
                        reward=reward_batch[i].item(),
                        next_state={
                            'hand_matrix': state_batch['hand_matrix'][i:i+1],
                            'discard_history': state_batch['discard_history'][i:i+1],
                            'valid_actions_mask': state_batch['valid_actions_mask'][i:i+1]
                        },
                        done=done_batch[i].item()
                    )
                
                # Optimize model
                agent.optimize_model()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': agent.current_loss or 0,
                    'memory': len(agent.memory)
                })
        
        # Save model checkpoint
        if (epoch + 1) % save_interval == 0:
            model_path = f"models/dqn_model_epoch_{epoch + 1}.pt"
            agent.save(model_path)
            print(f"Saved model checkpoint to {model_path}")
    
    # Save final model
    agent.save("models/dqn_model_final.pt")
    print("Training complete! Final model saved.")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    train_dqn() 