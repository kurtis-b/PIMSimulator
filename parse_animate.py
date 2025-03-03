import numpy as np
import re
import matplotlib.pyplot as plt

# Load the numpy arrays
DIM_IN = 1024
DIM_OUT = 4096
weight_array = np.round(np.load(f'data/gemv/gemv_weight_{DIM_OUT}x{DIM_IN}.npy'), 3)
output_array = np.round(np.load(f'data/gemv/gemv_output_{DIM_OUT}x{DIM_IN}.npy'), 3)
input_array = np.round(np.load(f'data/gemv/gemv_input_{DIM_OUT}x{DIM_IN}.npy'), 3)

# Initialize dictionaries to keep track of the ranges
ranges = {
    'ch': set(),
    'ra': set(),
    'bg': set(),
    'b': set(),
    'r': set(),
    'c': set()
}

def get_ranges(file_path):
    with open(file_path, 'r') as file:
        blocks = file.read().split("----------")
        for block in blocks:
            if "BANK_R" in block:
                if block.strip():
                    lines = block.split('\n')
                    # Process each line in the block
                    for line in lines:
                        # Extract the values for ch, ra, bg, b, r, and c
                        match = re.search(r'READ ch(\d+) ra(\d+) bg(\d+) b(\d+) r(\d+) c(\d+)\|\| \[\d+\] MAC GRF_B\[\d+\], GRF_A\[\d+\], (EVEN_BANK|ODD_BANK), auto @ \d+', line)
                        if match:
                            ranges['ch'].add(int(match.group(1)))
                            ranges['ra'].add(int(match.group(2)))
                            ranges['bg'].add(int(match.group(3)))
                            ranges['b'].add(int(match.group(4)))
                            ranges['r'].add(int(match.group(5)))
                            ranges['c'].add(int(match.group(6)))

def parse_pim_trace(file_path):
    channel_address_map = {}
    grf_a_maps = []

    with open(file_path, 'r') as file:
        blocks = file.read().split("----------")
        for block in blocks:
            if block.strip():
                grf_a_map = []
                lines = block.split('\n')
                # Check if the block is of a MAC command or something like a NOP
                mac_cmd_found = False
                for line in lines:
                    if "CMD" in line:
                        if "MAC" in line:
                            mac_cmd_found = True
                # Skip if no MAC command is found
                if not mac_cmd_found:
                    continue
                for line in lines:
                    # Extract the values for ch, ra, bg, b, r, and c
                    match = re.search(r'READ ch(\d+) ra(\d+) bg(\d+) b(\d+) r(\d+) c(\d+)\|\| \[\d+\] MAC GRF_B\[\d+\], GRF_A\[\d+\], (EVEN_BANK|ODD_BANK), auto @ \d+', line)
                    if match:
                        ch = int(match.group(1))
                        ra = int(match.group(2))
                        bg = int(match.group(3))
                        b = int(match.group(4))
                        r = int(match.group(5))
                        c = int(match.group(6))

                        if ch not in channel_address_map:
                            channel_address_map[ch] = {}
                        if ra not in channel_address_map[ch]:
                            channel_address_map[ch][ra] = {}
                        if bg not in channel_address_map[ch][ra]:
                            channel_address_map[ch][ra][bg] = {}
                        if b not in channel_address_map[ch][ra][bg]:
                            channel_address_map[ch][ra][bg][b] = {}
                        if r not in channel_address_map[ch][ra][bg][b]:
                            channel_address_map[ch][ra][bg][b][r] = {}
                        if c not in channel_address_map[ch][ra][bg][b][r]:
                            channel_address_map[ch][ra][bg][b][r][c] = {}

                    if "BANK_R" in line:
                        row, col = find_bank_match(line)
                        if row is not None:
                            if 'bank_match' not in channel_address_map[ch][ra][bg][b][r][c]:
                                channel_address_map[ch][ra][bg][b][r][c]['bank_match'] = (row, col)
                                # print(f"Found match at row {row}, col {col} for ch {ch}, ra {ra}, bg {bg}, b {b}, r {r}, c {c}")
                            else:
                                print(f"Match already found for ch {ch}, ra {ra}, bg {bg}, b {b}, r {r}, c {c}")
                        else:
                            print("No match found for line", line)

                    if "GRF_A" in line:
                        match_grf_a = re.search(r'\[GRF_A\]\[(\d+)\]\[(.*?)\]', line)
                        if match_grf_a:
                            key = int(match_grf_a.group(1))
                            values = list(map(lambda x: round(float(x), 3), match_grf_a.group(2).split()))
                            for idx in range(0, input_array.shape[1], 16):
                                if np.allclose(input_array[0][idx:idx+16], values, atol=0.01):
                                    grf_a_map.append(idx)
                                    break
                            else:
                                print(f"No GRF_A match found for GRF_A_{key} with values {values}")
                found_match = False
                for current_maps in grf_a_maps:
                    if np.allclose(current_maps['array_idxs'], grf_a_map):
                        current_maps['maps'].append([ch, ra, bg, b, r, c])
                        found_match = True
                        break
                if not found_match:
                    grf_a_maps.append({'array_idxs': grf_a_map, 'maps': [[ch, ra, bg, b, r, c]]})
    return channel_address_map, grf_a_maps

def find_bank_match(line):
    match = re.search(r'\[BANK_R\]\[\s*(.*?)\s*\]', line)
    if match:
        values = list(map(float, match.group(1).split()))
        for row in range(weight_array.shape[0]):
            for col in range(0, weight_array.shape[1], 16):
                if np.allclose(weight_array[row, col:col+16], values, atol=0.01):
                    return row, col
    return None, None

if __name__ == "__main__":
    # Get the ranges of the values
    get_ranges('pim_trace.out')

    # Print the ranges of the values
    for key, value_set in ranges.items():
        print(f"Range of {key}: {min(value_set)} to {max(value_set)}")

    # Print the shapes of the arrays to verify they are loaded correctly
    print(f"Weight array shape: {weight_array.shape}")
    print(f"Output array shape: {output_array.shape}")
    print(f"Input array shape: {input_array.shape}")

    # Parse the PIM trace file
    channel_address_map, grf_a_maps = parse_pim_trace('pim_trace.out')
    print(channel_address_map)
    print(grf_a_maps)