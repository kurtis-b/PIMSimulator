import sys
import os
from collections import OrderedDict

def count_substrings(file_path, unique_substrings):
    modes = ["HAB", "HAB mode", "HAB_PIM", "SB mode"]
    substring_count = OrderedDict((mode, {"invoke_count": 0, "substrings": {}}) for mode in modes)
    current_mode = "Setting up banks (Not a command)"
    substring_count[current_mode] = {"invoke_count": 0, "substrings": {}}
    modes_order = ["Setting up banks (Not a command)"]

    try:
        with open(file_path, 'r') as file:
            for line in file:
                command = line.split('ch')[0].strip()
                for mode in modes:
                    if mode == command and current_mode != command:
                        # print(f"Switching to mode: {mode}")
                        if mode not in modes_order:
                            modes_order.append(mode)
                        current_mode = mode
                        substring_count[current_mode]["invoke_count"] += 0
                        break
                if current_mode == command:
                    substring_count[current_mode]["invoke_count"] += 1
                else:
                    for substring in unique_substrings:
                        if substring == command:
                            if substring not in substring_count[current_mode]["substrings"]:
                                substring_count[current_mode]["substrings"][substring] = 0
                            substring_count[current_mode]["substrings"][substring] += 1
                # Check if a processing unit command is called--"[CMD]" will be in the line
                if "[CMD]" in line:
                    pu_cmd = "PU (not a command)"
                    start_idx = line.find('(') + 1
                    end_idx = line.find(')', start_idx)
                    if start_idx > 0 and end_idx > start_idx:
                        exact_cmd = line[start_idx:end_idx]
                    if pu_cmd not in substring_count[current_mode]["substrings"]:
                        substring_count[current_mode]["substrings"][pu_cmd] = {exact_cmd: 0}
                    else:
                        if exact_cmd not in substring_count[current_mode]["substrings"][pu_cmd]:
                            substring_count[current_mode]["substrings"][pu_cmd][exact_cmd] = 0
                    substring_count[current_mode]["substrings"][pu_cmd][exact_cmd] += 1
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    # Reorder the dictionary based on modes_order
    ordered_substring_count = OrderedDict()
    for mode in modes_order:
        ordered_substring_count[mode] = substring_count[mode]

    return ordered_substring_count

def parse_file(file_path):
    unique_substrings = set()
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'ch' in line:
                    substring = line.split('ch')[0].strip()
                    if substring:
                        unique_substrings.add(substring)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    return unique_substrings

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_mem_ctrl_cmds.py <file_path>")
    else:
        file_path = sys.argv[1]

        unique_substrings = parse_file(file_path)
        substring_count = count_substrings(file_path, unique_substrings)
        
        output_dir = os.path.splitext(file_path)[0]
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "mem_ctrl_cmd_counts.txt")
        with open(output_file_path, 'w') as output_file:
            output_file.write(f"Total counts of the commands executed after changing the mode:\n\n")
            for idx, key in enumerate(substring_count):
                output_file.write(f"{idx}--{key}:\n")
                output_file.write(f"  TOTAL TIMES INVOKED: {substring_count[key]['invoke_count']}\n")
                for substring in sorted(substring_count[key]["substrings"].keys()):
                    output_file.write(f"  {substring}: {substring_count[key]['substrings'][substring]}\n")
                output_file.write("\n")
