import json
import os
import random

def quick_verify():
    """Sample states from each file to estimate the number of unique games."""
    consolidated_dir = "../java/MavenProject/"
    
    print("Verifying consolidated training data files (quick check)...")
    
    for i in range(1, 11):
        file_path = f"{consolidated_dir}training_data_consolidated_{i}.json"
        
        if not os.path.exists(file_path):
            print(f"File {i}: Does not exist!")
            continue
            
        try:
            # Get file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print(f"File {i}: Size = {file_size:.2f} MB")
            
            # Check if the file is empty or too small
            if file_size < 0.1:
                print(f"File {i}: Too small, probably empty or corrupt")
                continue
                
            # Open and check the beginning of the file
            with open(file_path, 'r') as f:
                print(f"Checking file {file_path}...")
                # Read the first 100 characters to see if it's valid JSON
                start = f.read(100)
                if not start.strip().startswith('[') and not start.strip().startswith('{'):
                    print(f"File {i}: Not valid JSON")
                    continue
                
                # Reset file pointer
                f.seek(0)
                
                # Try to load the first few states
                try:
                    # Read the first part of the file
                    first_states_str = f.read(1000000)  # Read first ~1MB
                    # Find the last complete state
                    last_closing_brace = first_states_str.rfind('},')
                    if last_closing_brace > 0:
                        first_states_str = first_states_str[:last_closing_brace+1] + ']'
                    else:
                        # If no complete state, just use a valid JSON array
                        first_states_str = '[]'
                    
                    # Parse the partial content
                    first_states = json.loads(first_states_str.replace('}{', '},{'))
                    if first_states:
                        print(f"File {i}: Successfully parsed {len(first_states)} beginning states")
                        game_ids = set()
                        for state in first_states:
                            if isinstance(state, dict) and 'gameId' in state:
                                game_ids.add(state['gameId'])
                        print(f"File {i}: Found {len(game_ids)} unique games in the sample")
                except json.JSONDecodeError:
                    print(f"File {i}: Failed to parse beginning states")
                
                # Try counting lines as a rough estimate
                f.seek(0)
                line_count = 0
                for _ in range(1000):  # Count up to 1000 lines
                    line = f.readline()
                    if not line:
                        break
                    line_count += 1
                    if 'gameId' in line:
                        # Count occurrences of gameId
                        game_id_count = line.count('gameId')
                        if game_id_count > 0:
                            print(f"File {i}: Found gameId in lines")
                            break
                
                print(f"File {i}: Contains at least {line_count} lines of JSON")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("\nQuick verification complete.")
    print("To start training with these files, run either:")
    print("python train_dqn.py      # DQN training")
    print("python train_reinforce.py # REINFORCE training")

if __name__ == "__main__":
    quick_verify() 