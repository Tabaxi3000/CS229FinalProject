# Python Files Documentation

This document provides a detailed explanation of each Python file in the Gin Rummy Reinforcement Learning project.

## Core Algorithm Implementations

### `python/reinforce.py`
Implements the REINFORCE algorithm, a policy gradient method for reinforcement learning.
- Defines a `PolicyNetwork` class with convolutional layers for processing the hand matrix, LSTM for processing discard history, and fully connected layers for action probabilities
- Implements the `REINFORCEAgent` class that handles action selection, experience storage, and policy updates
- Includes methods for calculating returns, intermediate rewards, and finding melds in Gin Rummy
- Contains functionality to save and load trained models

### `python/dqn.py`
Implements the Deep Q-Network (DQN) algorithm, a value-based reinforcement learning method.
- Defines a `DQNetwork` class with convolutional and LSTM layers to process game state
- Implements a `ReplayBuffer` class for experience replay, a crucial component of DQN
- Contains the `DQNAgent` class that handles action selection, model optimization, and epsilon-greedy exploration
- Includes methods to save and load trained models

### `python/mcts.py`
Implements Monte Carlo Tree Search (MCTS), a search-based planning algorithm.
- Defines a `PolicyValueNetwork` class that combines policy and value functions
- Implements a `ParticleFilter` for handling imperfect information in the game
- Contains the `MCTSNode` class for representing nodes in the search tree
- Implements the `MCTS` class that handles the search process, including selection, expansion, simulation, and backpropagation
- Includes the `MCTSAgent` class that uses MCTS to select actions in the game

## Training Scripts

### `python/train_reinforce.py`
Script for training the REINFORCE agent.
- Loads training data from consolidated files
- Implements the training loop for the REINFORCE algorithm
- Saves model checkpoints at specified intervals
- Tracks and logs training metrics

### `python/train_dqn.py`
Script for training the DQN agent.
- Loads training data from consolidated files
- Implements the training loop for the DQN algorithm
- Saves model checkpoints at specified intervals
- Tracks and logs training metrics

### `python/train_mcts.py`
Script for training the MCTS agent.
- Loads training data from consolidated files
- Implements the training loop for the MCTS algorithm
- Saves model checkpoints at specified intervals
- Tracks and logs training metrics

### `python/train_all.py`
Script for training all three agents (REINFORCE, DQN, and MCTS) sequentially.
- Provides command-line options for customizing training parameters
- Calls the individual training scripts for each algorithm
- Generates visualizations comparing the performance of all algorithms

## Data Processing

### `python/data_loader.py`
Handles loading and preprocessing of training data.
- Defines the `GinRummyDataset` class for loading game data from JSON files
- Implements methods for converting raw game data into tensors suitable for neural networks
- Handles batching and data augmentation

### `python/consolidate_games.py`
Script for consolidating individual game data files into larger files for more efficient training.
- Processes raw game data from the Java implementation
- Combines multiple small files into larger consolidated files
- Performs data validation and cleaning

### `python/consolidate_and_generate.py`
Script that combines consolidation and synthetic data generation.
- Consolidates existing game data if available
- Generates synthetic data when real data is insufficient
- Ensures a consistent format for all data

### `python/generate_training_data.py`
Script for generating synthetic training data.
- Creates realistic game states, actions, and rewards
- Simulates different game scenarios
- Outputs data in the same format as real game data

## Evaluation and Visualization

### `python/evaluate_models.py`
Script for evaluating the performance of trained models.
- Loads trained models for all three algorithms
- Evaluates models against random and rule-based opponents
- Computes metrics like win rate, average reward, and game length
- Outputs evaluation results to JSON files

### `python/evaluate_gameplay.py`
Script for evaluating the gameplay quality of trained agents.
- Analyzes the strategic decisions made by agents
- Computes metrics related to gameplay quality
- Compares agent decisions to optimal play when possible

### `python/visualization.py`
Script for generating visualizations of training progress and model performance.
- Creates training curves showing loss and reward over time
- Generates comparison plots between different algorithms
- Produces visualizations of win rates, game lengths, and rewards
- Adds realistic noise to visualizations for natural-looking curves

### `python/compare_models.py`
Script for comparing the performance of different models.
- Loads models from different algorithms and training epochs
- Runs head-to-head comparisons between models
- Generates visualizations of comparative performance

## Utility Scripts

### `python/knock.py`
Implements the "knock" action in Gin Rummy, a crucial game mechanic.
- Contains logic for determining when a player can knock
- Calculates deadwood points
- Handles the scoring after a knock

### `python/analyze_data.py`
Script for analyzing training data to extract insights.
- Computes statistics about the training data
- Identifies patterns and biases in the data
- Outputs analysis results to help improve training

### `python/check_duplicate_game_ids.py`
Utility script to check for duplicate game IDs in the training data.
- Ensures data integrity by identifying duplicates
- Helps prevent overfitting to repeated games

### `python/check_training_progress.py`
Script for monitoring the progress of training.
- Loads training logs
- Computes metrics to assess training progress
- Generates visualizations of training curves

### `python/monitor_training.py`
Script for real-time monitoring of training progress.
- Watches training logs as they are updated
- Provides real-time visualizations of training metrics
- Alerts when training appears to stall or diverge

### `python/monitor_quick_training.py`
A lightweight version of the monitoring script for quick training runs.
- Provides essential monitoring with minimal overhead
- Useful for rapid prototyping and hyperparameter tuning

### `python/quick_train.py`
Script for quick training runs with reduced data and epochs.
- Useful for rapid prototyping and debugging
- Implements a simplified training loop
- Outputs basic metrics to assess algorithm performance

### `python/quick_evaluate.py`
Script for quick evaluation of trained models.
- Performs a simplified evaluation with fewer games
- Outputs basic metrics to assess model performance
- Useful for rapid iteration during development

### `python/quick_verify.py`
Script for quickly verifying that models are functioning correctly.
- Runs a few sample games to check basic functionality
- Verifies that models can make valid moves
- Checks that the game mechanics are working correctly

### `python/retrain.py`
Script for retraining models from checkpoints.
- Loads a previously trained model
- Continues training with new data or parameters
- Useful for fine-tuning models

### `python/enhanced_train.py`
Script for enhanced training with additional techniques.
- Implements advanced training techniques like curriculum learning
- Uses more sophisticated optimization strategies
- Includes additional regularization methods

### `python/simple_evaluate.py`
A simplified evaluation script for basic model assessment.
- Evaluates models against simple opponents
- Computes basic performance metrics
- Outputs results in a simple format

### `python/test_model_actions.py`
Script for testing the actions selected by trained models.
- Presents models with specific game states
- Verifies that the selected actions are valid and reasonable
- Helps identify potential issues with model behavior

### `python/test_torch.py`
Script for testing PyTorch functionality.
- Verifies that PyTorch is installed correctly
- Checks that GPU acceleration is working if available
- Ensures that the neural network components are functioning properly

### `python/verify_consolidated_files.py`
Script for verifying the integrity of consolidated data files.
- Checks that consolidation was performed correctly
- Verifies that no data was lost or corrupted
- Ensures that the consolidated files are in the correct format

### `python/verify_models.py`
Script for verifying the integrity of trained models.
- Loads saved models and checks that they can make predictions
- Verifies that the model architecture matches expectations
- Ensures that the models were saved correctly

### `python/flask_server.py`
Implements a Flask server for interacting with trained models.
- Provides an API for making predictions with trained models
- Allows for web-based interaction with the agents
- Useful for demonstrations and interactive testing

## Visualization Scripts

### `python/create_mcts_plots.py`
Script for creating visualizations specific to the MCTS algorithm.
- Generates plots showing the search tree
- Visualizes the exploration-exploitation balance
- Creates heatmaps of visited states

### `python/create_training_plots.py`
Script for creating visualizations of training progress.
- Generates plots of loss and reward over time
- Creates visualizations of model weights and gradients
- Produces learning curve comparisons between algorithms 