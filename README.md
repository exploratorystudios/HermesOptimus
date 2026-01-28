# Hermes Optimus: TI-84 Neural Network Word Classifier (V1.0)

## Project Overview

**Hermes Optimus** is an advanced word classification system for TI-84 Plus series graphing calculators. It features a TI-BASIC program that utilizes a sophisticated 4-input, 60-hidden neuron, 12-output neuron neural network. This network is trained on a PC for optimal performance and then its parameters (weights and biases) are transferred to the calculator as standard TI matrix and list files. This approach allows for robust classification of 4-letter words into one of twelve predefined categories, with a high tolerance for misspellings and input variations.

**Core Components:**

1.  **TI-BASIC Program (`HERMES1.8xp`):**
    * Performs word classification using the pre-loaded 4-60-12 neural network model.
    * Features a menu for classifying words and accessing information.
    * **Crucially, relies on the neural network model (matrices `[I]`, `[J]`, and lists `L₄`, `L₅`) being present in the calculator's RAM.** These files are created from CSVs using tools like SourceCoder/TokenIDE and then transferred by the user.

2.  **Python Scripts (for PC-based operations):**
    * **`train.py` (Trainer):** Trains the 4-60-12 neural network model. Outputs verbose TI-BASIC style weight assignments (for the CSV converter) and can save/load weights in NumPy's `.npz` format.
    * **`weights_to_csv.py` (CSV Exporter):** Converts the verbose weight file from the trainer into separate `.csv` files for matrices `[I]`, `[J]`, and lists `L₄`, `L₅`.
    * **`visualizer_script.py` (Visualizer):** Reads the verbose weight file to visualize the network's structure and letter-word activation associations.
    * *(The previous compression script is no longer used in this workflow.)*

**Target Word Categories (12 words):**
`BACK`, `DARK`, `EACH`, `FROM`, `JUST`, `BEEN`, `GOOD`, `MUCH`, `SOME`, `TIME`, `LIKE`, `ONLY`

## Core Functionality and Input Robustness

When trained extensively using `train.py` with its data augmentation features, Hermes Optimus exhibits significant robustness:

* **High Tolerance to Typos:** Often correctly classifies words despite multiple incorrect letters.
* **Anagram / Scrambled Letter Recognition:** Can frequently identify the correct category even with mixed letter order.
* **General Input Noise Immunity:** Resilient to various deviations from exact spellings.

The degree of this robustness is directly correlated with the comprehensiveness of the training performed by the Python script.

## Features

### TI-84 Program ("HERMES OPTIMUS V1.0")

* **Word Classification:** Classifies user-inputted 4-letter words into one of the 12 categories, displaying the top two predicted words and their confidence scores.
* **Direct Model Usage:** The program directly accesses matrices `[I]`, `[J]` and lists `L₄`, `L₅` from the calculator's RAM.
* **User Interface:** Menu-driven for easy navigation.
* **Neural Network Implementation:**
    * Architecture: 4-input, 60-hidden, 12-output feedforward neural network.
    * Activation Function: Sigmoid, with output clamping for stability.
* **On-Calculator Training (Vestigial - REMOVED from main flow):** Any on-calculator training code is considered legacy and is not part of the recommended workflow.

### Python Scripts

1.  **`train.py` (Trainer):**
    * Trains the 4-60-12 neural network.
    * Configurable learning rate and epochs.
    * Outputs verbose TI-BASIC style weight assignments (e.g., `nn_weights_4_60_12_verbose.txt`) which serve as input for `weights_to_csv.py`.
    * Can save/load weights in NumPy's `.npz` format for Python-side persistence and re-testing (e.g., `nn_model_4_60_12.npz`).
    * Implements data augmentation (numerical noise, word scrambling).
    * Includes a testing phase to evaluate model accuracy against ~40 scrambled words.

2.  **`weights_to_csv.py` (CSV Exporter):**
    * Reads the verbose TI-BASIC weight file from the trainer.
    * Outputs four `.csv` files: `prefix_weights_I.csv`, `prefix_weights_J.csv`, `prefix_bias_L4.csv`, `prefix_bias_L5.csv`. These are crucial for the new loading workflow.

3.  **`nn_visualizer_script.py` (Visualizer):**
    * Reads the verbose TI-BASIC weight file.
    * Visualizes network structure and letter-word activation heatmaps.

## Rationale for PC-Based Training & Direct Matrix/List File Loading

* **Computational Power:** Effective neural network training is best performed on a PC.
* **Data Management:** Instead of complex string compression and on-calculator decompression, this version uses a more robust method:
    1.  Weights are trained on PC.
    2.  Converted to CSV files.
    3.  CSVs are converted to standard TI Matrix (`.8xm`) and List (`.8xl`) files using tools like Cemetech's SourceCoder.
    4.  These native TI files, along with the main program, are transferred to the calculator.
* **Memory Efficiency:**
    * The main program file (`HERMES1.8xp`) is smaller as it no longer contains embedded data strings for weights. This removes the `ERR:MEMORY` *when loading the program/App itself*.
    * The calculator's RAM will be used to store the actual matrices `[I]`, `[J]` and lists `L₄`, `L₅` (approx. 12KB for a 4-60-12 network). This is more stable than manipulating very large strings in RAM.

## System Requirements

### TI-84 Program:

* **Calculator:** TI-84 Plus Silver Edition or compatible TI-84 Plus series model.
* **Transfer Software:** TI Connect™ CE software.
* **Calculator Memory:** Sufficient free RAM to hold the program, matrices `[I]` (~2.4KB), `[J]` (~8.6KB), lists `L₄` (~0.6KB), `L₅` (~0.1KB), and other operational variables. Total data ~17KB.

### Python Scripts:

* **Python Version:** Python 3.x.
* **Libraries:** NumPy, Matplotlib, NetworkX (`pip install numpy matplotlib networkx`).

## Setup and Installation

### TI-84 Program ("Hermes Optimus"):

1.  **Obtain Program:** Get the `HERMES1.8xp` file (this should have the simplified `Lbl P` that expects matrices/lists to be in RAM).
2.  **Prepare Matrix/List Files:** Follow **Steps 1-3** in the "Usage Instructions & Workflow" section below to generate `[I].8xm`, `[J].8xm`, `L4.8xl`, and `L5.8xl` files or download the pretrained versions provided with Hermes Optimus.
3.  **Transfer ALL Files:** Using TI Connect™ CE:
    * Click the "Calculator Explorer" icon (looks like two pages on the left toolbar).
    * Navigate to a desired location on your calculator in the right-hand pane (RAM or Archive).
    * Drag and drop `HERMES OPTIMUS.8xp`, `[I].8xm`, `[J].8xm`, `L4.8xl`, and `L5.8xl` from your computer into the TI Connect CE window for your calculator. Ensure all files are sent.

### Python Scripts:

1.  **Download:** Obtain `train.py`, `weights_to_csv.py`, and `visualizer_script.py`.
2.  **Install Python 3 & Libraries:** (As described in the previous README version).

## Usage Instructions & Workflow

This workflow details training, preparing data via CSVs and SourceCoder/TokenIDE, and deploying to the calculator.

**Step 1: Train the Neural Network Model (PC)**

* Use `train.py`.
* **Command Example:**
    ```bash
    python train.py --epochs 500000 --lr 0.01 --test
    ```
* **Key Output:** `nn_weights_4_60_12_verbose.txt` (and optionally `nn_model_4_60_12.npz`).

**Step 2: Convert Verbose Weights to CSV Files (PC)**

* Use `weights_to_csv.py`.
* **Command Example:**
    ```bash
    python weights_to_csv.py nn_weights_4_60_12_verbose.txt --output_dir hermes_csv_data --prefix hermes_model
    ```
* **Output:** This will create a directory (e.g., `hermes_csv_data`) containing:
    * `hermes_model_weights_I.csv`
    * `hermes_model_weights_J.csv`
    * `hermes_model_bias_L4.csv`
    * `hermes_model_bias_L5.csv`

**Step 3: Convert CSV Files to TI Calculator Files (PC - Using SourceCoder)**

1.  **Open SourceCoder:** This tools can import CSV data into TI data types.
    * **SourceCoder (Web):** Go to [Cemetech's SourceCoder](https://www.cemetech.net/sc/).
2.  **Import CSV and Export as TI Files:**
    * **For Matrix `[I]`:**
        * In SourceCoder/TokenIDE, find the option to create/import a matrix.
        * Import `hermes_model_weights_I.csv`.
        * Ensure dimensions are set correctly (60 rows, 4 columns for `[I]`).
        * Save/Export this matrix as `[I].8xm`.
    * **For Matrix `[J]`:**
        * Import `hermes_model_weights_J.csv`.
        * Ensure dimensions are set correctly (12 rows, 60 columns for `[J]`).
        * Save/Export this matrix as `[J].8xm`.
    * **For List `L₄`:**
        * Import `hermes_model_bias_L4.csv`.
        * Ensure it's a list with 60 elements.
        * Save/Export this list as `L4.8xl`. (The name on the calculator should be `L₄`).
    * **For List `L₅`:**
        * Import `hermes_model_bias_L5.csv`.
        * Ensure it's a list with 12 elements.
        * Save/Export this list as `L5.8xl`. (The name on the calculator should be `L₅`).
    * **Naming:** When saving/exporting from SourceCoder, ensure the filenames are exactly `[I].8xm`, `[J].8xm`, `L4.8xl`, and `L5.8xl` so the TI-BASIC program can find them by their default system names (`[I]`, `[J]`, `L₄`, `L₅`) once on the calculator.

**Step 4: Transfer Program and Data Files to Calculator**

1.  **Connect Calculator:** Connect your TI-84 to your PC.
2.  **Open TI Connect™ CE.**
3.  **Send Files:**
    * Click the "Calculator Explorer" icon (often on the left toolbar, may look like two pages).
    * Drag and drop the following files from your computer into the TI Connect CE window for your calculator (preferably into RAM):
        * `HERMES1.8xp` (your main program file)
        * `[I].8xm`
        * `[J].8xm`
        * `L4.8xl`
        * `L5.8xl`
    * Ensure all 5 files are successfully transferred.

**Step 5: Use Hermes Optimus on the Calculator**

1.  **Run Program:** On your TI-84, press `[prgm]`, select `HERMES1`, and press `[enter]`.
2.  **Main Menu:** Select `CONTINUE`.
3.  **Classifier Menu:**
    * `CLASSIFY`: Select this to input a 4-letter word. The program will use the matrices and lists directly from RAM.
    * `ABOUT`: Displays program information.
    * `EXIT`: Exits the program.

**Step 6 (Optional): Visualize or Further Analyze Weights on PC**

* Use `visualizer_script.py` with the command `python visualize.py nn_weights_4_60_12_verbose.txt --mode letter_heatmap`.

## Technical Specifications

* **Neural Network Architecture:** 4-60-12 (4 input neurons, 60 hidden layer neurons, 12 output neurons).
* **Word Categories:** 12 predefined 4-letter words.
* **Model Data Storage on Calculator:** Standard TI Matrix files (`[I].8xm`, `[J].8xm`) and TI List files (`L4.8xl`, `L5.8xl`) stored in RAM.
* **Activation Function:** Sigmoid.

## Troubleshooting & Notes

* **`ERR:MEMORY` on Calculator:**
    * If it occurs when **launching the App/program**: This is less likely now that large data strings are removed from the program text. Ensure your calculator has sufficient *initial* free RAM.
* **`ERR:UNDEFINED` when running `CLASSIFY`:** This means one or more of `[I]`, `[J]`, `L₄`, or `L₅` were not successfully transferred to the calculator or were deleted. Re-transfer the `.8xm` and `.8xl` files.
* **`ERR:INVALID DIM` during `CLASSIFY`:** This could indicate a mismatch between the dimensions expected by the TI-BASIC program (4-60-12) and the actual dimensions of the matrix/list files transferred. Ensure the CSV import and export process in SourceCoder used the correct dimensions.
* **Python Script Errors:** Ensure all Python dependencies are installed.
* **File Paths:** Use correct relative or absolute paths when running Python scripts.

This revised workflow leverages native TI file types for storing the model, which is generally more stable and efficient than handling very large custom-formatted strings for decompression.
