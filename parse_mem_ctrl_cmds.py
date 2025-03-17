import sys
import os
from collections import OrderedDict

ACTIVATE_KEY = "ACTIVATE"
CHANNEL_KEY = "ch"
CMD_CHANNEL_KEY = "ch[0]"
PU_CHANNEL_KEY = "ch0"
CMDS_KEY = "Commands after activation"
TAG_KEY = "Tag:"
INVOKES_KEY = "Total activations"

def count_substrings(file_path, unique_substrings):
    substring_count = []

    try:
        with open(file_path, 'r') as file:
            found_pu_cmd = False # Create a flag for checking subsequent lines for a PU command
            current_mode = '(No tag)'
            for line in file:
                # The first line in the trace file should be an ACTIVATE command
                # Will only use the trace from channel 0
                if ACTIVATE_KEY in line and CMD_CHANNEL_KEY in line: 
                    tag = line.split("tag :")
                    if tag[1].strip():
                        current_mode = tag[1].strip()
                    else:
                        current_mode = '(No tag)'
                    if len(substring_count) == 0:
                        substring_count.append({TAG_KEY: current_mode, INVOKES_KEY: 0, CMDS_KEY: {}})
                    elif substring_count[-1][TAG_KEY] != current_mode:
                        substring_count.append({TAG_KEY: current_mode, INVOKES_KEY: 0, CMDS_KEY: {}})
                    substring_count[-1][INVOKES_KEY] += 1
                else:
                    if PU_CHANNEL_KEY in line:
                        found_pu_cmd = True # Set the flag to check for a PU command in the subsequent lines
                    if CMD_CHANNEL_KEY in line:
                        command = line.split(CMD_CHANNEL_KEY)[0].strip()
                        for substring in unique_substrings:
                            if substring == command:
                                if substring not in substring_count[-1][CMDS_KEY]:
                                    substring_count[-1][CMDS_KEY][substring] = 0
                                substring_count[-1][CMDS_KEY][substring] += 1
                    elif found_pu_cmd:
                        # Check if a processing unit command is called--"[CMD]" will be in the line
                        if "[CMD]" in line:
                            pu_cmd = "PU (not a command)"
                            start_idx = line.find('(') + 1
                            end_idx = line.find(')', start_idx)
                            if start_idx > 0 and end_idx > start_idx:
                                exact_cmd = line[start_idx:end_idx]
                            if pu_cmd not in substring_count[-1][CMDS_KEY]:
                                substring_count[-1][CMDS_KEY][pu_cmd] = {exact_cmd: 0}
                            else:
                                if exact_cmd not in substring_count[-1][CMDS_KEY][pu_cmd]:
                                    substring_count[-1][CMDS_KEY][pu_cmd][exact_cmd] = 0
                            substring_count[-1][CMDS_KEY][pu_cmd][exact_cmd] += 1
                            found_pu_cmd = False # Reset the flag
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    
    return substring_count

def parse_file(file_path):
    unique_substrings = set()
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if CHANNEL_KEY in line:
                    substring = line.split(CHANNEL_KEY)[0].strip()
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
        total_activates = 0
        total_precharge = 0
        with open(output_file_path, 'w') as output_file:
            output_file.write(f"Total counts of the commands executed after an activation with each tag for 1 channel (2 banks and 1 pim unit):\n\n")
            for idx, activate_data in enumerate(substring_count):
                total_activates += activate_data[INVOKES_KEY]
                total_precharge += activate_data[CMDS_KEY]["PRECHARGE"] if "PRECHARGE" in activate_data[CMDS_KEY] else 0
            output_file.write(f"Total activates: {total_activates}\n")
            output_file.write(f"Total precharges: {total_precharge}\n\n")
            for idx, activate_data in enumerate(substring_count):
                cmd_description = f"{idx}--Activate tag: {activate_data[TAG_KEY]}"
                cmd_description += f" invoked {activate_data[INVOKES_KEY]} time"
                if activate_data[INVOKES_KEY] > 1:
                    cmd_description += "s"
                output_file.write(f"{cmd_description}\n")
                for substring in sorted(activate_data[CMDS_KEY].keys()):
                    output_file.write(f"  {substring}: {activate_data[CMDS_KEY][substring]}\n")
                output_file.write("\n")
                total_activates += activate_data[INVOKES_KEY]
                total_precharge += activate_data[CMDS_KEY]["PRECHARGE"] if "PRECHARGE" in activate_data[CMDS_KEY] else 0
        