import os
import time
import subprocess
import re

def check_training_progress():
    """Try to find and capture progress information from running training processes."""
    print("=== Training Progress Check ===")
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define pattern to look for epoch information
    epoch_pattern = r"Epoch (\d+)/(\d+)"
    
    # Try to capture output from the REINFORCE process
    try:
        print("\n=== REINFORCE Training Progress ===")
        result = subprocess.run(
            "ps aux | grep train_reinforce | grep -v grep",
            shell=True, capture_output=True, text=True
        )
        if "train_reinforce.py" in result.stdout:
            print("Process is running")
            # Try to find any log files or output files
            reinforce_logs = subprocess.run(
                "ls -la /tmp/reinforce_*.log 2>/dev/null || echo 'No log files found'",
                shell=True, capture_output=True, text=True
            )
            print(reinforce_logs.stdout)
        else:
            print("Process not found")
    except Exception as e:
        print(f"Error checking REINFORCE process: {e}")
    
    # Try to capture output from the DQN process
    try:
        print("\n=== DQN Training Progress ===")
        result = subprocess.run(
            "ps aux | grep train_dqn | grep -v grep",
            shell=True, capture_output=True, text=True
        )
        if "train_dqn.py" in result.stdout:
            print("Process is running")
            # Try to find any log files or output files
            dqn_logs = subprocess.run(
                "ls -la /tmp/dqn_*.log 2>/dev/null || echo 'No log files found'",
                shell=True, capture_output=True, text=True
            )
            print(dqn_logs.stdout)
        else:
            print("Process not found")
    except Exception as e:
        print(f"Error checking DQN process: {e}")
    
    # Check for any model files
    print("\n=== Model Files ===")
    try:
        model_files = subprocess.run(
            "ls -la models/ 2>/dev/null || echo 'No model files found'",
            shell=True, capture_output=True, text=True
        )
        print(model_files.stdout)
    except Exception as e:
        print(f"Error checking model files: {e}")
    
    # Check for memory usage
    print("\n=== Memory Usage ===")
    try:
        memory = subprocess.run(
            "ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -10",
            shell=True, capture_output=True, text=True
        )
        print(memory.stdout)
    except Exception as e:
        print(f"Error checking memory usage: {e}")

if __name__ == "__main__":
    check_training_progress() 