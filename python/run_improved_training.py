#!/usr/bin/env python3

import os
import sys
import subprocess

def main():
    """Run the improved training script with reward shaping."""
    # Check if python directory exists
    if not os.path.exists('python'):
        print("Error: 'python' directory not found. Make sure you're in the root directory.")
        sys.exit(1)
    
    # Check if the improved training files exist
    required_files = [
        'python/improved_gin_rummy_env.py',
        'python/improved_quick_train.py'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: '{file_path}' not found.")
            sys.exit(1)
    
    # Run the improved training script
    print("Starting improved training with reward shaping...")
    try:
        subprocess.run(['python3', 'python/improved_quick_train.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running improved training script: {e}")
        sys.exit(1)
    
    print("Improved training completed successfully!")
    print("\nThe improved training uses:")
    print("1. Reward shaping - providing immediate feedback for good actions")
    print("2. Prioritized experience replay - focusing on important experiences")
    print("3. Faster epsilon decay - more exploitation after initial exploration")
    print("4. Improved evaluation metrics - tracking rewards and win rates")
    
    print("\nTrained models are saved in the 'models' directory with the 'improved_' prefix.")

if __name__ == "__main__":
    main() 