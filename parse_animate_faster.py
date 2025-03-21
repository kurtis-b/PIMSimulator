import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import concurrent.futures
import json
import sys
import os

# Initialize dictionaries to keep track of the ranges
ranges = {
    'ch': set(),
    'ra': set(),
    'bg': set(),
    'b': set(),
    'r': set(),
    'c': set()
}

read_pattern = re.compile(r'READ ch(\d+) ra(\d+) bg(\d+) b(\d+) r(\d+) c(\d+)\|\| \[\d+\] MAC GRF_B\[\d+\], GRF_A\[\d+\], (EVEN_BANK|ODD_BANK)(, auto)? @ \d+')

def get_ranges(file_path):
    with open(file_path, 'r') as file:
        for block in file.read().split("----------"):
            if "BANK_R" in block and block.strip():
                for line in block.split('\n'):
                    match = read_pattern.search(line)
                    if match:
                        ranges['ch'].add(int(match.group(1)))
                        ranges['ra'].add(int(match.group(2)))
                        ranges['bg'].add(int(match.group(3)))
                        ranges['b'].add(int(match.group(4)))
                        ranges['r'].add(int(match.group(5)))
                        ranges['c'].add(int(match.group(6)))
                        break

def parse_pim_trace(file_path, log_ch, weight_array, input_array):
    channel_address_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))))
    grf_a_maps = []
    bank_r_pattern = re.compile(r'\[BANK_R\]\[\s*(.*?)\s*\]')
    grf_a_pattern = re.compile(r'\[GRF_A\]\[(\d+)\]\[(.*?)\]')

    with open(file_path, 'r') as file:
        found_start_mac = False
        for idx, block in enumerate(file.read().split("----------")):
            if block.strip():
                grf_a_map = []
                lines = block.split('\n')
                if any("NOP" in line for line in lines):
                    continue

                found_grf_a_cmd = False
                for line in lines:
                    if "GRF_A" in line:
                        found_grf_a_cmd = True
                    if not found_start_mac:
                        match = read_pattern.search(line)
                        if match:
                            ch, ra, bg, b, r, c = map(int, match.groups()[:6])
                            found_start_mac = True
                            if ch != log_ch:
                                break
                    else:
                        match = read_pattern.search(line)
                        if match:
                            ch, ra, bg, b, r, c = map(int, match.groups()[:6])
                            if ch != log_ch:
                                break

                        if "BANK_R" in line:
                            row, col = find_bank_match(line, bank_r_pattern, weight_array)
                            if row is not None:
                                channel_address_map[ch][ra][bg][b][r][c]['weight_array_idxs'] = (row, col)
                            else:
                                print(f"No BANK_R match found for ch{ch} ra{ra} bg{bg} b{b} r{r} c{c}")

                        found_grf_a_match = False
                        match_grf_a = grf_a_pattern.search(line)
                        if match_grf_a:
                            key = int(match_grf_a.group(1))
                            values = list(map(lambda x: float(x), match_grf_a.group(2).split()))
                            # print(f"GRF_A_{key} values: {values}")
                            for idx in range(0, input_array.shape[1], 16):
                                # print(f"Comparing {input_array[0][idx:idx+16]} with {values}")
                                # print(f"Difference: {np.abs(input_array[0][idx:idx+16] - values)}")
                                if np.allclose(input_array[0][idx:idx+16], values, atol=1e-2, rtol=0):
                                    grf_a_map.append(idx)
                                    found_grf_a_match = True
                                    break
                            if not found_grf_a_match:
                                print(f"No GRF_A match found for GRF_A_{key} with values {values}")
                if not found_grf_a_cmd:
                    print(f"No GRF_A command found in block {idx}")
                    continue # Skip trying to match current maps with grf_a_map since it wasn't found in the block

                found_match = False
                # Only consider the log channel
                if ch == log_ch: 
                    for current_maps in grf_a_maps:
                        if np.allclose(current_maps['in_vector_idxs'], grf_a_map):
                            current_maps['maps']['ch'].add(ch)
                            current_maps['maps']['ra'].add(ra)
                            current_maps['maps']['bg'].add(bg)
                            current_maps['maps']['b'].add(b)
                            current_maps['maps']['r'].add(r)
                            current_maps['maps']['c'].add(c)
                            found_match = True
                            break
                    if not found_match:
                        grf_a_maps.append({'in_vector_idxs': grf_a_map, 'maps': {'ch': {ch}, 'ra': {ra}, 'bg': {bg}, 'b': {b}, 'r': {r}, 'c': {c}}})
    return channel_address_map, grf_a_maps

def find_bank_match(line, bank_r_pattern, weight_array):
    match = bank_r_pattern.search(line)
    if match:
        values = list(map(lambda x: float(x), match.group(1).split()))
        for row in range(0, weight_array.shape[0]):
            for col in range(0, weight_array.shape[1], 16):
                if np.allclose(weight_array[row, col:col+16], values, atol=0.01):
                    return row, col
    return None, None

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python parse_animate_faster.py <log_ch> <file_name> <input_dim> <output_dim>")
        sys.exit(1)

    log_ch = int(sys.argv[1])
    file_name = str(sys.argv[2])
    dim_in = int(sys.argv[3])
    dim_out = int(sys.argv[4])

    weight_array = np.load(f'data/gemv/gemv_weight_{dim_out}x{dim_in}.npy')
    output_array = np.load(f'data/gemv/gemv_output_{dim_out}x{dim_in}.npy')
    input_array = np.load(f'data/gemv/gemv_input_{dim_out}x{dim_in}.npy')
    output_name = f'parse_output_ch{log_ch}.txt'

    # Get the ranges of the values
    get_ranges(file_name)

    # Create a folder using the name of the file to be parsed
    folder_name = os.path.splitext(file_name)[0]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    output_name = os.path.join(folder_name, f'parse_output_ch{log_ch}.txt')

    # Print the ranges of the values
    for key, value_set in ranges.items():
        print(f"Range of {key}: {min(value_set)} to {max(value_set)}")
    with open(output_name, 'w') as f:
        for key, value_set in ranges.items():
            f.write(f"Range of {key}: {min(value_set)} to {max(value_set)}\n")

    # Print the shapes of the arrays to verify they are loaded correctly
    print(f"Weight array shape: {weight_array.shape}")
    print(f"Output array shape: {output_array.shape}")
    print(f"Input array shape: {input_array.shape}")

    channel_address_map, grf_a_maps = parse_pim_trace(file_name, log_ch, weight_array, input_array)

    print(f"Log channel: {log_ch}")
    print(channel_address_map)
    print(grf_a_maps)

    # Write to a file
    with open(output_name, 'a') as f:
        f.write(f"Log channel: {log_ch}\n")
        f.write(json.dumps(channel_address_map, indent=4))
        f.write('\n')
        # Convert sets to lists for serialization
        for item in grf_a_maps:
            for key in item['maps']:
                item['maps'][key] = list(item['maps'][key])
        f.write(json.dumps(grf_a_maps, indent=4))
        f.write('\n')
