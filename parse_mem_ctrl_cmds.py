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
NO_TAG = "(No tag--setting up banks)"

def count_substrings(file_path, unique_substrings):
    substring_count = []

    try:
        with open(file_path, 'r') as file:
            found_pu_cmd = False # Create a flag for checking subsequent lines for a PU command
            current_mode = NO_TAG
            for line in file:
                # The first line in the trace file should be an ACTIVATE command
                # Will only use the trace from channel 0
                if ACTIVATE_KEY in line and CMD_CHANNEL_KEY in line: 
                    tag = line.split("tag :")
                    if tag[1].strip():
                        current_mode = tag[1].strip()
                    else:
                        current_mode = NO_TAG
                    if len(substring_count) == 0:
                        substring_count.append({TAG_KEY: current_mode, INVOKES_KEY: 0, CMDS_KEY: {}})
                    elif substring_count[-1][TAG_KEY] != current_mode:
                        substring_count.append({TAG_KEY: current_mode, INVOKES_KEY: 0, CMDS_KEY: {}})
                    substring_count[-1][INVOKES_KEY] += 1
                else:
                    if PU_CHANNEL_KEY in line:
                        command = line.split(PU_CHANNEL_KEY)[0].strip()
                        for substring in unique_substrings:
                            if substring == command:
                                if substring not in substring_count[-1][CMDS_KEY]:
                                    substring_count[-1][CMDS_KEY][substring] = 0
                                substring_count[-1][CMDS_KEY][substring] += 1
                        pu_cmd = "PU (not a command)"
                        start_idx = line.find("||") + 2
                        end_idx = line.find('@', start_idx)
                        if start_idx > 0 and end_idx > start_idx:
                            exact_cmd = line[start_idx:end_idx].strip()
                        if pu_cmd not in substring_count[-1][CMDS_KEY]:
                            substring_count[-1][CMDS_KEY][pu_cmd] = {exact_cmd: 0}
                        else:
                            if exact_cmd not in substring_count[-1][CMDS_KEY][pu_cmd]:
                                substring_count[-1][CMDS_KEY][pu_cmd][exact_cmd] = 0
                        substring_count[-1][CMDS_KEY][pu_cmd][exact_cmd] += 1
                    elif CMD_CHANNEL_KEY in line:
                        command = line.split(CMD_CHANNEL_KEY)[0].strip()
                        for substring in unique_substrings:
                            if substring == command:
                                if substring not in substring_count[-1][CMDS_KEY]:
                                    substring_count[-1][CMDS_KEY][substring] = 0
                                substring_count[-1][CMDS_KEY][substring] += 1
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
        substring_count = substring_count[1:] # Skip the first entry since it's setting up the banks
        
        output_dir = os.path.splitext(file_path)[0]
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "mem_ctrl_cmd_counts.txt")
        total_activates = 0
        total_precharge = 0
        total_writes = 0
        total_reads = 0
        total_macs = 0
        total_nops = 0
        total_jumps = 0

        total_rdpuall = 0
        total_comppuall = 0
        total_rdall = 0
        total_actbuf = 0
        total_actall = 0
        total_wrbuf = 0
        buffer_length = 16
        with open(output_file_path, 'w') as output_file:
            output_file.write(f"Total counts of the commands executed after an activation with each tag for 1 channel (2 banks and 1 pim unit):\n\n")
            for activate_data in substring_count:
                total_activates += activate_data[INVOKES_KEY]
                total_precharge += activate_data[CMDS_KEY]["PRECHARGE"] if "PRECHARGE" in activate_data[CMDS_KEY] else 0
                total_writes += activate_data[CMDS_KEY]["WRITE"] if "WRITE" in activate_data[CMDS_KEY] else 0
                total_reads += activate_data[CMDS_KEY]["READ"] if "READ" in activate_data[CMDS_KEY] else 0
                if "PU (not a command)" in activate_data[CMDS_KEY]:
                    for cmd, count in activate_data[CMDS_KEY]["PU (not a command)"].items():
                        if "MAC" in cmd:
                            total_macs += activate_data[CMDS_KEY]["PU (not a command)"][cmd]
                        elif "NOP" in cmd:
                            total_nops += activate_data[CMDS_KEY]["PU (not a command)"][cmd]
                        elif "JUMP" in cmd:
                            total_jumps += activate_data[CMDS_KEY]["PU (not a command)"][cmd]
                if "GRFB_TO_BANK_" in activate_data[TAG_KEY]:    
                    total_rdpuall += activate_data[CMDS_KEY]["WRITE"]
                elif "PROGRAM_CRFBAR" in activate_data[TAG_KEY]:
                    total_wrbuf += activate_data[CMDS_KEY]["BWRITE_CRF"] * buffer_length
                elif "MAC_" in activate_data[TAG_KEY]:
                    total_actall += activate_data[INVOKES_KEY]
                    for cmd, count in activate_data[CMDS_KEY]["PU (not a command)"].items():
                        if "MAC" in cmd:
                            total_comppuall += activate_data[CMDS_KEY]["PU (not a command)"][cmd]
                            total_rdall += activate_data[CMDS_KEY]["PU (not a command)"][cmd] * buffer_length
                    if "BWRITE_GRF_A" in activate_data[CMDS_KEY].keys():
                        total_wrbuf += activate_data[CMDS_KEY]["BWRITE_GRF_A"] * buffer_length
                elif "WRIO_TO_GRF_" in activate_data[TAG_KEY]:
                    if "PU (not a command)" in activate_data[CMDS_KEY].keys():
                        for cmd, count in activate_data[CMDS_KEY]["PU (not a command)"].items():
                            if "MAC" in cmd:
                                total_comppuall += activate_data[CMDS_KEY]["PU (not a command)"][cmd]
                                total_rdall += activate_data[CMDS_KEY]["PU (not a command)"][cmd] * buffer_length
                    total_wrbuf += activate_data[CMDS_KEY]["BWRITE_GRF_A"] * buffer_length
                    total_actbuf += activate_data[INVOKES_KEY]
                elif "PIMBAR" in activate_data[TAG_KEY]:
                    if "BWRITE_GRF_A" in activate_data[CMDS_KEY].keys():
                        total_wrbuf += activate_data[CMDS_KEY]["BWRITE_GRF_A"] * buffer_length
                    # elif "GRF_B_ZEROIZE" in activate_data[CMDS_KEY].keys():
                    #     total_wrbuf += activate_data[CMDS_KEY]["GRF_B_ZEROIZE"] * buffer_length
            output_file.write(f"Total activates: {total_activates}\n")
            output_file.write(f"Total precharges: {total_precharge}\n")
            output_file.write(f"Total writes: {total_writes}\n")
            output_file.write(f"Total reads: {total_reads}\n")
            output_file.write(f"Total macs: {total_macs}, Total jumps 7x: {total_jumps}, Total nops 8x: {total_nops}\n")
            output_file.write(f"Total PU executions: {total_macs + 7 * total_jumps + 8 * total_nops}\n\n")
            output_file.write(f"Total RD_PU_ALL: {total_rdpuall}\n")
            output_file.write(f"Total COMP_PU_ALL: {total_comppuall}\n")
            output_file.write(f"Total RD_ALL: {total_rdall}\n")
            output_file.write(f"Total ACT_BUF: {total_actbuf}\n")
            output_file.write(f"Total ACT_ALL: {total_actall}\n")
            output_file.write(f"Total WR_BUF: {total_wrbuf}\n\n")
            for idx, activate_data in enumerate(substring_count): 
                cmd_description = f"{idx+1}--Activate tag: {activate_data[TAG_KEY]}"
                cmd_description += f" invoked {activate_data[INVOKES_KEY]} time"
                if activate_data[INVOKES_KEY] > 1:
                    cmd_description += "s"
                output_file.write(f"{cmd_description}\n")
                for substring in sorted(activate_data[CMDS_KEY].keys()):
                    output_file.write(f"  {substring}: {activate_data[CMDS_KEY][substring]}\n")
                output_file.write("\n")
        