#!/bin/bash

# List of values to pass to the script
# values=(0 1 2 3 8 9 16 17 24 25 32 33 40 41 48 49 56 57 60 61 62 63)
values=(48 49 56 57 60 61 62 63)

# Loop through each value and run the script in the background
for value in "${values[@]}"; do
    python3 parse_animate_faster.py "$value" &
done

# Wait for all background processes to finish
wait

echo "All processes have completed."
