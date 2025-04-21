import os
import re
from itertools import product

# Input files and their respective modifications
input_files = [
    {
        "file": "3DDRAM_Samsung3D8Gb_extened.cfg",
        "modifications": {
            "-UCA bank count": [4, 8, 16, 32],
            "-page size (bits)": [1024, 2048, 4096, 8192, 16384, 32768], # in bits
        },
    },
    {
        "file": "2DDRAM_Samsung2GbDDR2.cfg",
        "modifications": {
            "-UCA bank count": [4, 8, 16, 32],
            "-page size (bits)": [1024, 2048, 4096, 8192, 16384, 32768], # in bits
        },
    },
    {
        "file": "ddr3_cache.cfg",
        "modifications": {
            "-size (bytes)": [32, 64, 128, 256, 512, 1024, 2048], # in bytes
        },
    },
]

# Output directory
output_dir = "generated_configs"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each input file
for file_info in input_files:
    input_file = file_info["file"]
    modifications = file_info["modifications"]

    # Ensure the input file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"The file '{input_file}' does not exist.")

    # Read the content of the input file
    with open(input_file, "r") as file:
        content = file.readlines()

    # Generate all combinations of modifications
    keys = list(modifications.keys())
    values = list(modifications.values())
    combinations = list(product(*values))

    # Loop through each combination and create new config files
    for i, combination in enumerate(combinations):
        modified_content = []
        for line in content:
            modified_line = line
            for key, value in zip(keys, combination):
                if key in line:
                    print(f"Modifying line: {line.strip()} with {key} {value}")
                    modified_line = re.sub(rf"{re.escape(key)}\s+\d+", f"{key} {value}", line)
                    print(f"Modified line: {modified_line.strip()}")
                    break
            modified_content.append(modified_line)

        # Write the modified content to a new file
        output_file = os.path.join(output_dir, f"{os.path.basename(input_file).split('.')[0]}_config_{i + 1}.cfg")
        with open(output_file, "w") as file:
            file.writelines(modified_content)

print(f"Config files have been created in the '{output_dir}' directory.")
