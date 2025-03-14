import os
import time
import subprocess

def check_process_status(process_name):
    """Check if a process is running and get its CPU/memory usage."""
    cmd = f"ps -p $(pgrep -f {process_name}) -o pid,%cpu,%mem,etime,comm 2>/dev/null || echo 'Process not running'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def check_model_files():
    """Check for model checkpoint files."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return "Models directory does not exist yet."
    
    files = os.listdir(models_dir)
    if not files:
        return "No model files found yet."
    
    return f"Found {len(files)} model files: {', '.join(files)}"

def main():
    """Monitor training progress."""
    print("=== Training Monitor ===")
    print("\nChecking process status:")
    print("\nREINFORCE:")
    print(check_process_status("train_reinforce.py"))
    print("\nDQN:")
    print(check_process_status("train_dqn.py"))
    
    print("\nChecking model files:")
    print(check_model_files())
    
    print("\nCurrent time:", time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Check the processes for CPU spikes (active training)
    reinforce_active = "Process not running" not in check_process_status("train_reinforce.py")
    dqn_active = "Process not running" not in check_process_status("train_dqn.py")
    
    if reinforce_active and dqn_active:
        print("\nBoth REINFORCE and DQN training are still active!")
    elif reinforce_active:
        print("\nOnly REINFORCE training is still active. DQN may have finished or crashed.")
    elif dqn_active:
        print("\nOnly DQN training is still active. REINFORCE may have finished or crashed.")
    else:
        print("\nNeither training process is running. Both have finished or crashed.")

if __name__ == "__main__":
    main() 