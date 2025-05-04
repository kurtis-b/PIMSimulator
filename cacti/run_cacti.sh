#!/bin/bash

# Directory containing the .cfg files
CONFIG_DIR="generated_configs"

# Check if the directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Directory $CONFIG_DIR does not exist."
    exit 1
fi

# Loop through each .cfg file in the directory
for cfg_file in "$CONFIG_DIR"/*.cfg; do
    # Check if there are any .cfg files
    if [ ! -e "$cfg_file" ]; then
        echo "No .cfg files found in $CONFIG_DIR."
        exit 1
    fi

    # Run the cacti executable with the .cfg file as input and save output to a log file
    log_file="${cfg_file%.cfg}.out"
    echo "Running cacti with $cfg_file... Output will be saved to $log_file"
    # Ensure the script has write permissions for the directory and the log file
    if [ ! -w "$CONFIG_DIR" ]; then
        echo "Cannot write to directory $CONFIG_DIR. Check permissions."
        exit 1
    fi
    touch "$log_file" 2>/dev/null || { echo "Cannot write to $log_file. Check permissions."; exit 1; }
    ./cacti -infile "$cfg_file" > "$log_file" 2>&1
done