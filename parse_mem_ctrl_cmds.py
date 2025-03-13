import sys
import os

def count_substrings(file_path, unique_substrings):
    substring_count = {substring: 0 for substring in unique_substrings}
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                for substring in unique_substrings:
                    if substring in line:
                        substring_count[substring] += 1
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    return substring_count

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
            for substring in sorted(substring_count.keys()):
                output_file.write(f"{substring}: {substring_count[substring]}\n")   
