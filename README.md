# Quantum Historical Data Search

This repository contains Greg and Victor's final project for CSCI 300: Quantum Computing at St. Olaf College.

## Project Overview

This project demonstrates the power of quantum computing through two main applications:

1. **Quantum Machine Learning (QML)**: Using Variational Quantum Classifiers to predict gladiator survival based on historical data
2. **Quantum Optimization**: Using the Quantum Approximate Optimization Algorithm (QAOA) to find optimal groupings of historical Wikipedia topics

## Installation

### Option 1: Using Conda (Recommended)

If you have conda installed, you can use the provided environment file:

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate quantum_computing

# Install required packages in the conda environment
pip install -r requirements.txt
```

**Important**: Make sure you activate the conda environment before installing packages or running the notebook!

### Option 2: Using pip (if not using conda)

If you prefer using pip without conda, install the required packages:

```bash
pip install -r requirements.txt
```

### Required Packages

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning utilities
- `pennylane` - Quantum machine learning framework
- `pennylane-lightning` - High-performance quantum simulator

## Running the Code

### Using Jupyter Notebook

1. **Activate the conda environment** (if using conda):
   ```bash
   conda activate quantum_computing
   ```

2. Make sure you have Jupyter installed in the environment:
   ```bash
   pip install jupyter
   ```

3. **Important**: Make sure the Jupyter kernel is using the correct environment:
   ```bash
   # Install ipykernel in the conda environment
   pip install ipykernel
   
   # Register the kernel (if needed)
   python -m ipykernel install --user --name quantum_computing --display-name "Python (quantum_computing)"
   ```

4. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

5. Open `main.ipynb` and **select the correct kernel** (Kernel → Change Kernel → Python (quantum_computing))

6. Run all cells (Cell → Run All)

**Troubleshooting**: If you get "No module named 'matplotlib'" errors:
- Make sure the notebook kernel is set to use the `quantum_computing` conda environment
- Restart the kernel (Kernel → Restart Kernel)
- Verify packages are installed: In a notebook cell, run `!pip list | grep matplotlib`

### Using Python Script

Alternatively, you can convert the notebook to a Python script and run it:

```bash
# Convert notebook to Python script (if needed)
jupyter nbconvert --to script main.ipynb

# Run the script
python main.py
```

## Data Files

The project uses two datasets:

- `gladiator_data.csv` - Historical gladiator data with features like age, wins, losses, and survival status
- `wiki_data.csv` - Wikipedia entries about historical terms and concepts

Make sure these files are in the same directory as the notebook before running.

## Project Structure

```
quantum-historical-data-search/
├── main.ipynb              # Main Jupyter notebook with all code
├── main.py                 # Python script version (if converted)
├── gladiator_data.csv      # Gladiator dataset
├── wiki_data.csv           # Wikipedia historical dataset
├── coins_data.parquet      # Coin dataset (not used in main project)
├── requirements.txt        # Python package dependencies
├── environment.yml         # Conda environment file
└── README.md              # This file
```

## Features

### Part 1: Quantum Machine Learning
- Variational Quantum Classifier (VQC) implementation
- Feature encoding using angle encoding
- Training with hybrid quantum-classical optimization
- Classification accuracy evaluation

### Part 2: Quantum Optimization
- QAOA implementation for Max-Cut problem
- Graph construction from topic similarities
- Optimal partition finding
- Visualization of results

## Notes

- The code uses PennyLane's default quantum simulator. For real quantum hardware, you would need to configure a quantum device provider (e.g., IBM Quantum, IonQ).
- Training the quantum classifier may take several minutes depending on your hardware.
- The QAOA optimization is set to 50 iterations by default, which may take a few minutes to complete.

## Documentation

For a comprehensive beginner's guide explaining every piece of code in detail, see:

📖 **[QUANTUM_COMPUTING_GUIDE.md](QUANTUM_COMPUTING_GUIDE.md)** - Complete educational guide covering:
- Quantum computing basics (qubits, superposition, entanglement)
- Detailed explanation of every code section
- Classical ML concepts (KNN, SVM)
- Quantum optimization (QAOA) step-by-step
- Glossary of terms
- Common questions and answers

This guide is designed for beginners learning quantum computing and provides detailed explanations of all concepts and code.

## Authors

Greg and Victor - CSCI 300: Quantum Computing, St. Olaf College