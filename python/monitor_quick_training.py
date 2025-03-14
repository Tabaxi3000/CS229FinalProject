#!/usr/bin/env python3
import os
import subprocess
import time
import glob
import datetime

def check_process_status():
    """Check if quick_train.py is running."""
    cmd = "ps aux | grep '[p]ython3 quick_train.py' | wc -l"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    count = int(result.stdout.strip())
    
    if count > 0:
        print("✅ quick_train.py is running")
        
        # Get more details about the process
        cmd = "ps aux | grep '[p]ython3 quick_train.py'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        process_info = result.stdout.strip()
        
        # Extract CPU and memory usage
        parts = process_info.split()
        if len(parts) >= 3:
            cpu_usage = parts[2] if len(parts) > 2 else "N/A"
            mem_usage = parts[3] if len(parts) > 3 else "N/A"
            start_time = parts[9] if len(parts) > 9 else "N/A"
            
            print(f"   CPU usage: {cpu_usage}%")
            print(f"   Memory usage: {mem_usage}%")
            print(f"   Process start time: {start_time}")
    else:
        print("❌ quick_train.py is not running")

def check_model_files():
    """Check for model checkpoint files."""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print("❌ Models directory does not exist yet.")
        return
    
    # Check for model files
    dqn_files = glob.glob(os.path.join(models_dir, "dqn_quick*.pt"))
    reinforce_files = glob.glob(os.path.join(models_dir, "reinforce_quick*.pt"))
    
    if dqn_files:
        print(f"✅ Found {len(dqn_files)} DQN model files:")
        for file in sorted(dqn_files):
            size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
            timestamp = os.path.getmtime(file)
            time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   - {os.path.basename(file)} ({size:.2f} MB) - Last modified: {time_str}")
    else:
        print("❌ No DQN model files found yet.")
    
    if reinforce_files:
        print(f"✅ Found {len(reinforce_files)} REINFORCE model files:")
        for file in sorted(reinforce_files):
            size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
            timestamp = os.path.getmtime(file)
            time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   - {os.path.basename(file)} ({size:.2f} MB) - Last modified: {time_str}")
    else:
        print("❌ No REINFORCE model files found yet.")

def check_mini_dataset():
    """Check if the mini dataset was created."""
    mini_file = "../java/MavenProject/mini_training_data.json"
    
    if os.path.exists(mini_file):
        size = os.path.getsize(mini_file) / (1024 * 1024)  # Convert to MB
        timestamp = os.path.getmtime(mini_file)
        time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(f"✅ Mini dataset exists: {mini_file} ({size:.2f} MB) - Created: {time_str}")
    else:
        print("❌ Mini dataset has not been created yet.")

def estimate_progress():
    """Try to estimate the overall progress based on file timestamps."""
    mini_file = "../java/MavenProject/mini_training_data.json"
    models_dir = "models"
    
    if not os.path.exists(mini_file):
        print("❓ Cannot estimate progress: Mini dataset not found")
        return
    
    mini_file_time = os.path.getmtime(mini_file)
    now = time.time()
    
    # Expected total training time based on the script (REINFORCE: 5 epochs * 200 iterations, DQN: same)
    expected_total_time = 30 * 60  # 30 minutes in seconds
    
    # Check if any model files exist
    dqn_files = glob.glob(os.path.join(models_dir, "dqn_quick*.pt"))
    reinforce_files = glob.glob(os.path.join(models_dir, "reinforce_quick*.pt"))
    
    if not (dqn_files or reinforce_files):
        # Only mini dataset exists
        elapsed = now - mini_file_time
        if elapsed < expected_total_time:
            progress = elapsed / expected_total_time * 100
            print(f"⏳ Estimated overall progress: {progress:.1f}% (Still in dataset creation or early training)")
        else:
            print("⚠️ Training may be taking longer than expected.")
        return
    
    # If we have REINFORCE files but no DQN files, we're in the middle
    if reinforce_files and not dqn_files:
        last_reinforce_time = max(os.path.getmtime(f) for f in reinforce_files)
        elapsed = now - mini_file_time
        progress = 50.0  # Halfway point - completed REINFORCE, starting DQN
        print(f"⏳ Estimated overall progress: {progress:.1f}% (Completed REINFORCE training, waiting for DQN)")
        return
    
    # If we have both, check most recent
    if reinforce_files and dqn_files:
        last_file_time = max(max(os.path.getmtime(f) for f in reinforce_files), 
                            max(os.path.getmtime(f) for f in dqn_files))
        
        # If the most recent file is the final model, training is complete
        if "final.pt" in max(dqn_files, key=os.path.getmtime):
            print("✅ Training appears to be complete! Final models have been saved.")
            return
        
        # Otherwise, estimate based on number of epoch files
        reinforce_epochs = len([f for f in reinforce_files if "epoch" in f])
        dqn_epochs = len([f for f in dqn_files if "epoch" in f])
        max_epochs = 5  # From the quick_train.py script
        
        if dqn_epochs > 0:
            progress = (reinforce_epochs + dqn_epochs) / (max_epochs * 2) * 100
            print(f"⏳ Estimated overall progress: {progress:.1f}% (REINFORCE: {reinforce_epochs}/{max_epochs} epochs, DQN: {dqn_epochs}/{max_epochs} epochs)")
        else:
            progress = reinforce_epochs / (max_epochs * 2) * 100
            print(f"⏳ Estimated overall progress: {progress:.1f}% (REINFORCE: {reinforce_epochs}/{max_epochs} epochs)")

def main():
    """Main monitoring function."""
    print("=== QUICK TRAINING MONITOR ===")
    print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n-- Process Status --")
    check_process_status()
    
    print("\n-- Mini Dataset --")
    check_mini_dataset()
    
    print("\n-- Model Files --")
    check_model_files()
    
    print("\n-- Progress Estimate --")
    estimate_progress()
    
    print("\n=== END OF MONITORING REPORT ===")

if __name__ == "__main__":
    main() 