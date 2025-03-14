#!/bin/bash

# Navigate to the MavenProject directory
cd "$(dirname "$0")"

# Compile the project
echo "Compiling the project..."
mvn clean compile

# Run the data collection program
echo "Starting data collection to generate the remaining games..."
mvn exec:java -Dexec.mainClass="collector.DataCollectionMain"

echo "Data collection process completed or terminated."
echo "Check the generated files with the pattern 'training_data_remaining_*.json'" 