import re
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import networkx as nx
from string import ascii_uppercase

# --- Configuration: Network Dimensions (MUST MATCH THE WEIGHTS FILE) ---
# For the 4-60-12 architecture
DEFAULT_INPUT_SIZE = 4
DEFAULT_HIDDEN_SIZE = 60
DEFAULT_OUTPUT_SIZE = 12
DEFAULT_CATEGORIES = [ # For 4-60-12 architecture
    "BACK", "DARK", "EACH", "FROM", "JUST", "BEEN", "GOOD", "MUCH", 
    "SOME", "TIME", "LIKE", "ONLY"
]
# ---

def sigmoid(x):
    """Sigmoid activation function with clipping for stability."""
    x = np.clip(x, -20, 20) # Increased clip range slightly for activation
    return 1 / (1 + np.exp(-x))

def encode_char_for_input(char_to_encode, position, input_size):
    """Encodes a single character at a specific position in an input vector."""
    input_vector = np.zeros(input_size)
    if 'A' <= char_to_encode <= 'Z' and 0 <= position < input_size:
        input_vector[position] = (ord(char_to_encode) - ord('A') + 1) / 26.0
    return input_vector

def forward_pass(inputs, w_i, b_l4, w_j, b_l5):
    """Performs a forward pass using provided weights and biases."""
    if w_i is None or b_l4 is None or w_j is None or b_l5 is None:
        print("Error: Weights or biases not loaded for forward pass.")
        return np.zeros(b_l5.shape[0] if b_l5 is not None else DEFAULT_OUTPUT_SIZE) # Return zeros if weights missing

    # Hidden layer
    hidden_inputs_raw = np.dot(w_i, inputs) + b_l4
    hidden_outputs = sigmoid(hidden_inputs_raw)
    
    # Output layer
    final_inputs_raw = np.dot(w_j, hidden_outputs) + b_l5
    final_outputs = sigmoid(final_inputs_raw)
    
    return final_outputs

def parse_weights_from_ti_file(filepath, expected_input_size, expected_hidden_size, expected_output_size):
    """
    Parses weights and biases from the verbose TI-BASIC assignment file.
    """
    print(f"Parsing weights for architecture: {expected_input_size}-{expected_hidden_size}-{expected_output_size}")
    
    weights_i_matrix = np.zeros((expected_hidden_size, expected_input_size))
    weights_j_matrix = np.zeros((expected_output_size, expected_hidden_size))
    bias_l4_list = np.zeros(expected_hidden_size)
    bias_l5_list = np.zeros(expected_output_size)
    
    parsed_i_count, parsed_j_count, parsed_l4_count, parsed_l5_count = 0, 0, 0, 0
    expected_i_total = expected_hidden_size * expected_input_size
    expected_j_total = expected_output_size * expected_hidden_size
    expected_l4_total = expected_hidden_size
    expected_l5_total = expected_output_size
    expected_total_all = expected_i_total + expected_j_total + expected_l4_total + expected_l5_total

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line_content in enumerate(f, 1):
                line_content = line_content.strip()
                if line_content.startswith(":") or \
                   line_content.startswith("{") or \
                   "dim(" in line_content or \
                   "→L₄" in line_content and "{" in line_content or \
                   "→L₅" in line_content and "{" in line_content or \
                   not "→" in line_content or \
                   not line_content:
                    continue # Skip comments, initializations, non-assignments
                
                parts = line_content.split("→")
                if len(parts) != 2: continue
                
                val_str = parts[0].replace("⁻", "-")
                target_expr = parts[1]
                try: value = float(val_str)
                except ValueError: 
                    print(f"Warning: Could not parse float from '{val_str}' on line {line_num}")
                    continue

                m_i = re.match(r"\[I\]\((\d+),(\d+)\)", target_expr)
                m_j = re.match(r"\[J\]\((\d+),(\d+)\)", target_expr)
                m_l4 = re.match(r"L₄\((\d+)\)", target_expr)
                m_l5 = re.match(r"L₅\((\d+)\)", target_expr)
                
                if m_i:
                    r, c_idx = int(m_i.group(1)) - 1, int(m_i.group(2)) - 1
                    if 0 <= r < expected_hidden_size and 0 <= c_idx < expected_input_size: 
                        weights_i_matrix[r, c_idx] = value; parsed_i_count += 1
                elif m_j:
                    r, c_idx = int(m_j.group(1)) - 1, int(m_j.group(2)) - 1
                    if 0 <= r < expected_output_size and 0 <= c_idx < expected_hidden_size: 
                        weights_j_matrix[r, c_idx] = value; parsed_j_count +=1
                elif m_l4:
                    idx = int(m_l4.group(1)) - 1
                    if 0 <= idx < expected_hidden_size: 
                        bias_l4_list[idx] = value; parsed_l4_count +=1
                elif m_l5:
                    idx = int(m_l5.group(1)) - 1
                    if 0 <= idx < expected_output_size: 
                        bias_l5_list[idx] = value; parsed_l5_count +=1
    except FileNotFoundError:
        print(f"Error: Input file '{filepath}' not found.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None, None, None, None

    total_parsed_successfully = parsed_i_count + parsed_j_count + parsed_l4_count + parsed_l5_count
    print(f"\nParsing Summary:")
    print(f"  Parsed for [I]: {parsed_i_count}/{expected_i_total}")
    print(f"  Parsed for [J]: {parsed_j_count}/{expected_j_total}")
    print(f"  Parsed for L4:  {parsed_l4_count}/{expected_l4_total}")
    print(f"  Parsed for L5:  {parsed_l5_count}/{expected_l5_total}")
    print(f"  Total Parsed:   {total_parsed_successfully}/{expected_total_all}")

    if total_parsed_successfully != expected_total_all:
        print(f"Warning: Total parsed values ({total_parsed_successfully}) does NOT match total expected ({expected_total_all}).")
    
    return weights_i_matrix, weights_j_matrix, bias_l4_list, bias_l5_list

def visualize_network_structure(input_size, hidden_size, output_size, weights_i=None, weights_j=None):
    """Visualizes the basic layered structure of the neural network."""
    G = nx.DiGraph()
    input_nodes = [f"I{i+1}" for i in range(input_size)]
    hidden_nodes = [f"H{i+1}" for i in range(hidden_size)]
    output_nodes = [f"O{i+1}" for i in range(output_size)]

    for i, node in enumerate(input_nodes): G.add_node(node, layer=0, label=node)
    for i, node in enumerate(hidden_nodes): G.add_node(node, layer=1, label=node)
    for i, node in enumerate(output_nodes): G.add_node(node, layer=2, label=node)

    for i_idx, i_node in enumerate(input_nodes):
        for h_idx, h_node in enumerate(hidden_nodes):
            G.add_edge(i_node, h_node)
    for h_idx, h_node in enumerate(hidden_nodes):
        for o_idx, o_node in enumerate(output_nodes):
            G.add_edge(h_node, o_node)

    pos = {}
    max_nodes_in_layer = max(input_size, hidden_size, output_size)
    y_scaling_factor = 0.8 # To reduce vertical spread if too many nodes
    
    for i, node in enumerate(input_nodes): pos[node] = (0, (max_nodes_in_layer - input_size) * y_scaling_factor / 2 + i * y_scaling_factor)
    for i, node in enumerate(hidden_nodes): pos[node] = (1, (max_nodes_in_layer - hidden_size) * y_scaling_factor / 2 + i * y_scaling_factor)
    for i, node in enumerate(output_nodes): pos[node] = (2, (max_nodes_in_layer - output_size) * y_scaling_factor / 2 + i * y_scaling_factor)

    fig_width = 10 
    fig_height = max(8, max_nodes_in_layer * 0.25) # Adjust height based on max layer size

    plt.figure(figsize=(fig_width, fig_height))
    node_colors = ['skyblue'] * input_size + ['lightgreen'] * hidden_size + ['salmon'] * output_size
    
    # Adjust node size and font size based on the number of nodes to avoid clutter
    node_size_val = max(50, 20000 / max_nodes_in_layer**1.1) if max_nodes_in_layer > 0 else 500
    font_size_val = max(4, 10 - max_nodes_in_layer * 0.05) if max_nodes_in_layer > 0 else 8

    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), 
            node_size=node_size_val, node_color=node_colors, font_size=font_size_val,
            width=0.3, arrowsize=8, font_weight='bold')
    plt.title(f"Neural Network Structure ({input_size}-{hidden_size}-{output_size})", fontsize=14)
    plt.tight_layout()
    plt.show()

def visualize_letter_word_heatmap(input_size, hidden_size, output_size, 
                                  weights_i, weights_j, bias_l4, bias_l5, 
                                  categories):
    """
    Visualizes the activation of output words for each letter at each input position.
    """
    if weights_i is None or weights_j is None or bias_l4 is None or bias_l5 is None:
        print("Error: Full weights and biases are required for letter-word activation heatmap.")
        return

    alphabet = ascii_uppercase # A-Z
    num_letters = len(alphabet)
    
    # activation_matrix[letter_idx * input_size + pos_idx, word_idx]
    activation_matrix = np.zeros((num_letters * input_size, output_size))
    
    y_labels = [] # For heatmap y-axis

    print("\nCalculating activations for letter-position heatmap...")
    for letter_idx, letter in enumerate(alphabet):
        for pos_idx in range(input_size):
            input_vector = encode_char_for_input(letter, pos_idx, input_size)
            output_activations = forward_pass(input_vector, weights_i, bias_l4, weights_j, bias_l5)
            
            row_idx = letter_idx * input_size + pos_idx
            activation_matrix[row_idx, :] = output_activations
            y_labels.append(f"{letter} @ Pos{pos_idx+1}")
            if row_idx % 20 == 0: # Progress update
                 print(f"  Processed {row_idx+1} / {num_letters * input_size} letter-position inputs...")
    print("Activation calculation complete.")

    fig, ax = plt.subplots(figsize=(max(10, output_size * 0.8), max(12, num_letters * input_size * 0.2)))
    cax = ax.imshow(activation_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(y_labels)))
    
    ax.set_xticklabels(categories, rotation=90, ha="right")
    ax.set_yticklabels(y_labels)
    
    # Add a colorbar
    fig.colorbar(cax, label='Activation Strength')
    
    ax.set_title('Letter-Position vs. Word Activation Heatmap', fontsize=16)
    ax.set_xlabel('Output Words (Categories)', fontsize=12)
    ax.set_ylabel('Input Letter @ Position', fontsize=12)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a neural network structure or letter-word activations from a TI-BASIC verbose weights file.")
    parser.add_argument("input_ti_weights_file", 
                        help="Path to the verbose TI-BASIC weight file.")
    parser.add_argument("--mode", type=str, default="structure", choices=["structure", "letter_heatmap"],
                        help="Type of visualization: 'structure' or 'letter_heatmap' (default: structure).")
    
    parser.add_argument("--input_neurons", type=int, default=DEFAULT_INPUT_SIZE, 
                        help=f"Number of input neurons (default: {DEFAULT_INPUT_SIZE})")
    parser.add_argument("--hidden_neurons", type=int, default=DEFAULT_HIDDEN_SIZE, 
                        help=f"Number of hidden neurons (default: {DEFAULT_HIDDEN_SIZE})")
    parser.add_argument("--output_neurons", type=int, default=DEFAULT_OUTPUT_SIZE, 
                        help=f"Number of output neurons (default: {DEFAULT_OUTPUT_SIZE})")

    args = parser.parse_args()

    # Determine categories based on output_neurons, or use default if they match
    # For simplicity, we'll assume DEFAULT_CATEGORIES matches DEFAULT_OUTPUT_SIZE
    # A more robust script might load categories from a file or another argument.
    current_categories = DEFAULT_CATEGORIES
    if args.output_neurons != len(DEFAULT_CATEGORIES):
        print(f"Warning: Output neurons ({args.output_neurons}) does not match default categories ({len(DEFAULT_CATEGORIES)}). Using generic O1, O2... labels for heatmap.")
        current_categories = [f"O{i+1}" for i in range(args.output_neurons)]


    print(f"Visualizing network architecture: {args.input_neurons}-{args.hidden_neurons}-{args.output_neurons}")

    weights_i, weights_j, bias_l4, bias_l5 = parse_weights_from_ti_file(
        args.input_ti_weights_file,
        args.input_neurons, # Note: parse_weights expects hidden, input, output
        args.hidden_neurons,
        args.output_neurons
    )

    if args.mode == "structure":
        if weights_i is not None and weights_j is not None: # weights_i and weights_j are optional for structure plot
            visualize_network_structure(args.input_neurons, args.hidden_neurons, args.output_neurons, weights_i, weights_j)
        else: # Still try to plot structure even if weights parsing had issues
             print("Attempting to visualize structure without detailed weight information due to parsing issues...")
             visualize_network_structure(args.input_neurons, args.hidden_neurons, args.output_neurons)
    elif args.mode == "letter_heatmap":
        if weights_i is None or weights_j is None or bias_l4 is None or bias_l5 is None:
            print("Error: Cannot generate letter_heatmap because weights and biases could not be fully parsed.")
        else:
            visualize_letter_word_heatmap(args.input_neurons, args.hidden_neurons, args.output_neurons,
                                          weights_i, weights_j, bias_l4, bias_l5,
                                          current_categories)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
