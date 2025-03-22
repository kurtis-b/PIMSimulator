#!/bin/bash

# List of values to pass to the script
values=(0 1 2 3 8 9 15 16 17 24 25 32 33 40 41 48 49 56 57 60 61 62 63)
file_name="pim_trace_smaller.out"
input_dim=256
output_dim=128

# Loop through each value and run the script in the background
for value in "${values[@]}"; do
    python3 parse_output_channels.py "$value" "$file_name" "$input_dim" "$output_dim" &
done

# Wait for all background processes to finish
wait

echo "All processes have completed."
