import re
import numpy as np
import argparse
import os

# Network Dimensions (MUST MATCH THE SOURCE WEIGHTS FILE)
# These should match the architecture the verbose weights file was generated for.
# For the 4-60-12 architecture:
INPUT_SIZE = 4
HIDDEN_SIZE = 60 
OUTPUT_SIZE = 12

def parse_weights_from_ti_file(filepath, expected_hidden_size, expected_input_size, expected_output_size):
    """
    Parses weights and biases from the verbose TI-BASIC assignment file.
    Args:
        filepath (str): Path to the verbose TI-BASIC weight file.
        expected_hidden_size (int): Expected number of hidden neurons.
        expected_input_size (int): Expected number of input neurons.
        expected_output_size (int): Expected number of output neurons.
    Returns:
        tuple: (weights_i_matrix, weights_j_matrix, bias_l4_list, bias_l5_list)
               Returns None if parsing fails significantly.
    """
    # Initialize with correct dimensions
    weights_i_matrix = np.zeros((expected_hidden_size, expected_input_size))
    weights_j_matrix = np.zeros((expected_output_size, expected_hidden_size))
    bias_l4_list = np.zeros(expected_hidden_size)
    bias_l5_list = np.zeros(expected_output_size)
    
    parsed_i_count = 0
    parsed_j_count = 0
    parsed_l4_count = 0
    parsed_l5_count = 0

    expected_i_total = expected_hidden_size * expected_input_size
    expected_j_total = expected_output_size * expected_hidden_size
    expected_l4_total = expected_hidden_size
    expected_l5_total = expected_output_size
    expected_total_all = expected_i_total + expected_j_total + expected_l4_total + expected_l5_total

    print(f"Expecting {expected_i_total} for [I], {expected_j_total} for [J], {expected_l4_total} for L4, {expected_l5_total} for L5.")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line_content in enumerate(f, 1):
            line_content = line_content.strip()
            # Skip comments, empty lines, and structural lines
            if line_content.startswith(":") or \
               line_content.startswith("{") or \
               line_content.startswith("Lbl P_VERBOSE_LOAD") or \
               line_content.startswith("ClrHome") or \
               line_content.startswith("Output(1,1,\"LOADING VERBOSE WEIGHTS\")") or \
               line_content.startswith("Output(3,1,\"VERBOSE LOAD DONE\")") or \
               line_content.startswith("Pause") or \
               line_content.startswith("Goto A") or \
               not line_content:
                continue
            
            parts = line_content.split("→")
            if len(parts) != 2: 
                print(f"Skipping malformed line {line_num}: {line_content}")
                continue
            
            val_str = parts[0].replace("⁻", "-") # TI-BASIC negative to standard
            target_expr = parts[1]

            try: 
                value = float(val_str)
            except ValueError: 
                print(f"Warning: Could not parse float from '{val_str}' on line {line_num}")
                continue

            # Regex for different assignment types
            m_i = re.match(r"\[I\]\((\d+),(\d+)\)", target_expr)
            m_j = re.match(r"\[J\]\((\d+),(\d+)\)", target_expr)
            m_l4 = re.match(r"L₄\((\d+)\)", target_expr) # TI uses subscript ₄
            m_l5 = re.match(r"L₅\((\d+)\)", target_expr) # TI uses subscript ₅
            
            parsed_this_line = False
            if m_i:
                r, c_idx = int(m_i.group(1)) - 1, int(m_i.group(2)) - 1 # Adjust to 0-based index
                if 0 <= r < expected_hidden_size and 0 <= c_idx < expected_input_size: 
                    weights_i_matrix[r, c_idx] = value # Use NumPy array indexing
                    parsed_i_count += 1
                    parsed_this_line = True
                else:
                    print(f"Warning: Index out of bounds for [I] on line {line_num}: {target_expr}")
            elif m_j:
                r, c_idx = int(m_j.group(1)) - 1, int(m_j.group(2)) - 1 # Adjust to 0-based index
                if 0 <= r < expected_output_size and 0 <= c_idx < expected_hidden_size: 
                    weights_j_matrix[r, c_idx] = value # Use NumPy array indexing
                    parsed_j_count +=1
                    parsed_this_line = True
                else:
                    print(f"Warning: Index out of bounds for [J] on line {line_num}: {target_expr}")
            elif m_l4:
                idx = int(m_l4.group(1)) - 1 # Adjust to 0-based index
                if 0 <= idx < expected_hidden_size: 
                    bias_l4_list[idx] = value
                    parsed_l4_count +=1
                    parsed_this_line = True
                else:
                    print(f"Warning: Index out of bounds for L4 on line {line_num}: {target_expr}")
            elif m_l5:
                idx = int(m_l5.group(1)) - 1 # Adjust to 0-based index
                if 0 <= idx < expected_output_size: 
                    bias_l5_list[idx] = value
                    parsed_l5_count +=1
                    parsed_this_line = True
                else:
                    print(f"Warning: Index out of bounds for L5 on line {line_num}: {target_expr}")
            
            if not parsed_this_line and "→" in line_content: # If it looked like an assignment but wasn't parsed
                 print(f"Warning: Unparsed assignment-like line {line_num}: {line_content}")

    total_parsed_successfully = parsed_i_count + parsed_j_count + parsed_l4_count + parsed_l5_count
    print(f"\nParsing Summary:")
    print(f"  Parsed for [I]: {parsed_i_count}/{expected_i_total}")
    print(f"  Parsed for [J]: {parsed_j_count}/{expected_j_total}")
    print(f"  Parsed for L4:  {parsed_l4_count}/{expected_l4_total}")
    print(f"  Parsed for L5:  {parsed_l5_count}/{expected_l5_total}")
    print(f"  Total Parsed:   {total_parsed_successfully}/{expected_total_all}")

    if total_parsed_successfully != expected_total_all:
        print(f"Warning: Total parsed values ({total_parsed_successfully}) does NOT match total expected ({expected_total_all}).")
        print("         The output CSV files might be incomplete or incorrect.")
        # You might want to return None or raise an error here if strictness is required
    
    return weights_i_matrix, weights_j_matrix, bias_l4_list, bias_l5_list

def save_to_csv(data, filename, output_dir="."):
    """
    Saves a NumPy array or list to a CSV file.
    Args:
        data (numpy.ndarray or list): The data to save.
        filename (str): The name of the CSV file.
        output_dir (str): The directory to save the CSV file in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    # Ensure data is a NumPy array for savetxt
    if not isinstance(data, np.ndarray):
        data_np = np.array(data)
    else:
        data_np = data

    # For 1D arrays (vectors like L4, L5), reshape to be a column vector for savetxt
    if data_np.ndim == 1:
        data_np = data_np.reshape(-1, 1)
        
    np.savetxt(filepath, data_np, delimiter=",", fmt='%.7f') # Save with ~7 decimal places
    print(f"Successfully saved data to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Convert TI-BASIC verbose weights file to CSV files.")
    parser.add_argument("input_ti_weights_file", 
                        help="Path to the verbose TI-BASIC weight file (e.g., nn_weights_4_60_12_verbose.txt).")
    parser.add_argument("--output_dir", type=str, default="csv_weights",
                        help="Directory to save the output CSV files (default: 'csv_weights').")
    parser.add_argument("--prefix", type=str, default="nn_4_60_12",
                        help="Prefix for the output CSV filenames (default: 'nn_4_60_12').")
    
    # Add arguments for network dimensions to make the script more flexible
    parser.add_argument("--input_neurons", type=int, default=INPUT_SIZE, help=f"Number of input neurons (default: {INPUT_SIZE})")
    parser.add_argument("--hidden_neurons", type=int, default=HIDDEN_SIZE, help=f"Number of hidden neurons (default: {HIDDEN_SIZE})")
    parser.add_argument("--output_neurons", type=int, default=OUTPUT_SIZE, help=f"Number of output neurons (default: {OUTPUT_SIZE})")

    args = parser.parse_args()

    print(f"Using network architecture: {args.input_neurons}-{args.hidden_neurons}-{args.output_neurons}")

    weights_i, weights_j, bias_l4, bias_l5 = parse_weights_from_ti_file(
        args.input_ti_weights_file,
        args.hidden_neurons,
        args.input_neurons,
        args.output_neurons
    )

    if weights_i is not None: # Check if parsing returned data
        save_to_csv(weights_i, f"{args.prefix}_weights_I.csv", args.output_dir)
        save_to_csv(weights_j, f"{args.prefix}_weights_J.csv", args.output_dir)
        save_to_csv(bias_l4,   f"{args.prefix}_bias_L4.csv",   args.output_dir)
        save_to_csv(bias_l5,   f"{args.prefix}_bias_L5.csv",   args.output_dir)
        print("\nCSV file generation complete.")
    else:
        print("\nCSV file generation aborted due to parsing issues.")

if __name__ == "__main__":
    main()
