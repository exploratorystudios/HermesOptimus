import numpy as np
import random
import argparse
import os

class TICalcNeuralNetwork:
    def __init__(self, learning_rate=0.01, epochs=30000, hidden_size=60, output_size=12, input_size=4): # ARCHITECTURE UPDATE
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size 
        
        self.I = (np.random.rand(self.hidden_size, self.input_size) - 0.5) / 2
        self.J = (np.random.rand(self.output_size, self.hidden_size) - 0.5) / 4 
        self.L4 = np.zeros(self.hidden_size)
        self.L5 = np.zeros(self.output_size) 

        # Updated categories list for 12 words
        self.categories = [
            "BACK", "DARK", "EACH", "FROM", "JUST", "BEEN", "GOOD", "MUCH", 
            "SOME", "TIME", "LIKE", "ONLY"
        ]
        if len(self.categories) != self.output_size:
            raise ValueError(f"Number of categories ({len(self.categories)}) in the script must match output_size ({self.output_size})")

    def sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-x))
    
    def encode_word(self, word):
        input_values = np.zeros(self.input_size)
        word = word.upper()[:self.input_size]
        for i, char_val in enumerate(word): # Renamed char to char_val to avoid conflict
            if 'A' <= char_val <= 'Z':
                input_values[i] = (ord(char_val) - ord('A') + 1) / 26.0
        return input_values
    
    def forward(self, inputs):
        hidden_inputs_raw = np.dot(self.I, inputs) + self.L4
        hidden_outputs = self.sigmoid(hidden_inputs_raw)
        final_inputs_raw = np.dot(self.J, hidden_outputs) + self.L5
        final_outputs = self.sigmoid(final_inputs_raw)
        return hidden_outputs, final_outputs

    def train(self, word_str, category_index, augment=True):
        targets = np.zeros(self.output_size)
        targets[category_index] = 1.0
        repeats = 5 if augment else 1
        
        for i_repeat in range(repeats):
            current_word_str = word_str
            if augment and (i_repeat == 1 or i_repeat == 2): 
                list_word = list(word_str)
                random.shuffle(list_word)
                current_word_str = "".join(list_word)
            
            inputs = self.encode_word(current_word_str)

            if augment and (i_repeat == 3 or i_repeat == 4): 
                noise_application_prob = 0.5 
                for k in range(self.input_size):
                    if random.random() < noise_application_prob:
                         if inputs[k] > 0 or k < len(current_word_str): 
                            inputs[k] = random.random() / 5.0

            hidden_outputs, final_outputs = self.forward(inputs)
            
            delta_output = (final_outputs - targets) * final_outputs * (1.0 - final_outputs)
            error_propagated_to_hidden = np.dot(self.J.T, delta_output)
            delta_hidden = error_propagated_to_hidden * hidden_outputs * (1.0 - hidden_outputs)
            
            self.J -= self.learning_rate * np.outer(delta_output, hidden_outputs)
            self.L5 -= self.learning_rate * delta_output
            self.I -= self.learning_rate * np.outer(delta_hidden, inputs)
            self.L4 -= self.learning_rate * delta_hidden
            
    def train_model(self, verbose=True):
        training_data = []
        for i, category_name in enumerate(self.categories):
            word_to_train = category_name[:self.input_size] 
            training_data.append((word_to_train, i))

        if verbose:
            print(f"Starting training for {self.epochs} epochs with LR={self.learning_rate}, Arch: {self.input_size}-{self.hidden_size}-{self.output_size}")
        
        for epoch in range(self.epochs):
            random.shuffle(training_data)
            for word, category_idx in training_data:
                self.train(word, category_idx, augment=True)
            
            if verbose and (epoch + 1) % (self.epochs // 20 if self.epochs >=20 else 1) == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} completed")
        
        if verbose:
            print("Training complete!")
    
    def predict(self, word):
        inputs = self.encode_word(word)
        _, outputs = self.forward(inputs)
        category_index = np.argmax(outputs)
        sum_outputs = np.sum(outputs)
        confidence = 0
        if sum_outputs > 0: 
            confidence = int((outputs[category_index] / sum_outputs) * 100)
        if 0 <= category_index < len(self.categories):
            return self.categories[category_index], confidence
        else:
            return "ERR_PREDICT_IDX", 0

    def test_accuracy(self, test_words_map=None, num_scrambled_target=24): # Target ~2 scrambles per category for 12 categories
        if test_words_map is None:
            print(f"Generating default test map aiming for ~{num_scrambled_target} unique scrambled words...")
            test_words_map = {}
            scrambled_words_generated = 0
            for word_cat in self.categories:
                test_words_map[word_cat] = word_cat
            
            num_categories = len(self.categories)
            scrambles_per_cat_target = (num_scrambled_target // num_categories if num_categories > 0 else 0) + 1
            attempts_per_category_heuristic = scrambles_per_cat_target * 3 + 5 
            
            for word_cat in self.categories:
                if scrambled_words_generated >= num_scrambled_target: break
                unique_scrambles_for_this_cat = 0
                if len(word_cat) <= 1: continue
                for _ in range(attempts_per_category_heuristic):
                    char_list = list(word_cat)
                    random.shuffle(char_list)
                    scrambled = "".join(char_list)
                    if scrambled != word_cat and scrambled not in test_words_map:
                        test_words_map[scrambled] = word_cat
                        scrambled_words_generated += 1
                        unique_scrambles_for_this_cat +=1
                        if unique_scrambles_for_this_cat >= scrambles_per_cat_target or \
                           scrambled_words_generated >= num_scrambled_target:
                            break 
            print(f"Generated test map with {len(test_words_map)} total entries ({scrambled_words_generated} actual scrambled unique words added).")

        correct_predictions = 0
        total_testable_words = len(test_words_map)
        if total_testable_words == 0:
            print("\nNo test words for accuracy calculation."); return []
        
        results_log = []
        print("\n--- Testing Model Accuracy ---")
        for word_input, expected_category_word in test_words_map.items():
            predicted_category_word, confidence = self.predict(word_input)
            is_match = (predicted_category_word == expected_category_word)
            if is_match: correct_predictions += 1
            result_str = f"Input: '{word_input}' → Predicted: '{predicted_category_word}' ({confidence}%) - Expected: '{expected_category_word}' [{ 'MATCH' if is_match else 'MISS' }]"
            results_log.append(result_str); print(result_str)
        accuracy = (correct_predictions / total_testable_words) * 100 if total_testable_words > 0 else 0
        print(f"\nAccuracy on this test set: {accuracy:.2f}% ({correct_predictions}/{total_testable_words})")
        print("--- End of Test ---")
        return results_log

    def generate_verbose_ti_basic_weights(self):
        l4_zeros = "{" + ",".join(["0"]*self.hidden_size) + "}"
        l5_zeros = "{" + ",".join(["0"]*self.output_size) + "}" 
        ti_basic_code = f""": Verbose Weights for TI-BASIC (Input for compress_weights.py)
: Network Architecture: {self.input_size}-{self.hidden_size}-{self.output_size}
{{{self.hidden_size},{self.input_size}}}→dim([I])
{{{self.output_size},{self.hidden_size}}}→dim([J])
{l4_zeros}→L₄
{l5_zeros}→L₅
"""
        for r in range(self.hidden_size):
            for c_idx in range(self.input_size): # Renamed c to c_idx
                val_str = f"{self.I[r, c_idx]:.6f}".replace("-", "⁻")
                ti_basic_code += f"{val_str}→[I]({r+1},{c_idx+1})\n"
        for r in range(self.output_size):
            for c_idx in range(self.hidden_size): # Renamed c to c_idx
                val_str = f"{self.J[r, c_idx]:.6f}".replace("-", "⁻")
                ti_basic_code += f"{val_str}→[J]({r+1},{c_idx+1})\n"
        for i, val in enumerate(self.L4):
            val_str = f"{val:.6f}".replace("-", "⁻")
            ti_basic_code += f"{val_str}→L₄({i+1})\n"
        for i, val in enumerate(self.L5):
            val_str = f"{val:.6f}".replace("-", "⁻")
            ti_basic_code += f"{val_str}→L₅({i+1})\n"
        return ti_basic_code

    def save_verbose_weights_to_file(self, filename="nn_weights_verbose.txt"):
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_verbose_ti_basic_weights())
        print(f"Verbose TI-BASIC weights saved to {filename}")

    def save_numpy_weights(self, filename="nn_weights.npz"):
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        np.savez_compressed(filename, i_weights=self.I, j_weights=self.J, l4_biases=self.L4, l5_biases=self.L5)
        print(f"NumPy weights and biases saved to {filename}")

    def load_numpy_weights(self, filename="nn_weights.npz"):
        if not os.path.exists(filename):
            print(f"Error: Weight file '{filename}' not found."); return False
        try:
            data = np.load(filename)
            if not ('i_weights' in data and 'j_weights' in data and 'l4_biases' in data and 'l5_biases' in data):
                print(f"Error: Weight file '{filename}' missing required arrays."); return False
            if data['i_weights'].shape != (self.hidden_size, self.input_size) or \
               data['j_weights'].shape != (self.output_size, self.hidden_size) or \
               data['l4_biases'].shape != (self.hidden_size,) or \
               data['l5_biases'].shape != (self.output_size,):
                print(f"Error: Weight dimensions in '{filename}' mismatch network ({self.input_size}-{self.hidden_size}-{self.output_size})."); return False
            self.I = data['i_weights']; self.J = data['j_weights']
            self.L4 = data['l4_biases']; self.L5 = data['l5_biases']
            print(f"NumPy weights and biases loaded from {filename}"); return True
        except Exception as e:
            print(f"Error loading weights from '{filename}': {e}"); return False

def main():
    parser = argparse.ArgumentParser(description='Train NN (4-60-12), save/load NumPy weights, test.')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--load_weights', type=str, metavar='<npz_file>', help='Load .npz weights, skip training.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01).')
    parser.add_argument('--epochs', type=int, default=4000000, help='Training epochs (default: 50000).') # Increased for potentially more complex task
    # hidden_size and output_size are now fixed in the class for this version
    parser.add_argument('--output_verbose_ti', type=str, default="nn_weights_4_60_12_verbose.txt", metavar='<txt_file>', help='Output for verbose TI-BASIC weights.')
    parser.add_argument('--save_numpy_weights', type=str, default="nn_weights_4_60_12.npz", metavar='<npz_file>', help='Output for NumPy weights (if training).')
    parser.add_argument('--test', action='store_true', help='Run tests after training or loading.')
    parser.add_argument('--no-verbose-train', action='store_true', help='Suppress training progress.')
    args = parser.parse_args()
    
    nn = TICalcNeuralNetwork(learning_rate=args.lr, epochs=args.epochs, hidden_size=60, output_size=12) 
    
    if args.load_weights:
        if not nn.load_numpy_weights(args.load_weights): print("Exiting due to load failure."); return
        print("Weights loaded. Skipping training.")
    else:
        print("Starting training...")
        nn.train_model(verbose=not args.no_verbose_train)
        if args.output_verbose_ti: nn.save_verbose_weights_to_file(args.output_verbose_ti)
        if args.save_numpy_weights: nn.save_numpy_weights(args.save_numpy_weights)
            
    if args.test:
        print("\nProceeding to testing phase...")
        nn.test_accuracy(num_scrambled_target=24) # Target ~24 scrambled for 12 categories

if __name__ == "__main__":
    main()