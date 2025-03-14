import json
import os
from collections import defaultdict

def verify_consolidated_files():
    """Verify that each consolidated file contains 10,000 games."""
    consolidated_dir = "../java/MavenProject/"
    all_good = True
    
    print("Verifying consolidated training data files...")
    
    for i in range(1, 11):
        file_path = f"{consolidated_dir}training_data_consolidated_{i}.json"
        
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} does not exist!")
            all_good = False
            continue
            
        try:
            print(f"Checking file {i}...")
            
            # Get file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print(f"  File size: {file_size:.2f} MB")
            
            with open(file_path, 'r') as f:
                print("  Loading JSON...")
                data = json.load(f)
                
                # Extract game states
                if isinstance(data, list):
                    states = data
                else:
                    states = data.get('gameStates', [])
                
                print(f"  Counting unique games in {len(states)} states...")
                
                # Count unique game IDs with progress updates
                game_ids = set()
                for idx, state in enumerate(states):
                    if idx % 100000 == 0 and idx > 0:
                        print(f"  Processed {idx}/{len(states)} states ({idx/len(states)*100:.1f}%)...")
                    
                    game_id = state.get('gameId', 0)
                    game_ids.add(game_id)
                
                total_states = len(states)
                unique_games = len(game_ids)
                
                if unique_games == 10000:
                    print(f"✅ File {i}: {unique_games} games ({total_states} states) - CORRECT")
                else:
                    print(f"❌ File {i}: {unique_games} games ({total_states} states) - WRONG (expected 10000)")
                    all_good = False
                
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")
            all_good = False
    
    # Summary
    if all_good:
        print("\n✅ All files contain exactly 10,000 games each. Ready for training!")
    else:
        print("\n⚠️ Some files do not contain 10,000 games. Please fix before training.")
    
    return all_good

if __name__ == "__main__":
    verify_consolidated_files() 