#!/bin/bash

# Create necessary directories
mkdir -p models metrics plots

# Run a quick training session with fewer episodes for testing
python train_all_models.py --episodes 500 --eval-interval 50 --save-interval 100 --models all

echo "Quick training complete! Check the plots directory for learning curves." 