# Quantum Computing Project: Complete Guide

**Authors:** Greg and Victor  
**Course:** CSCI 300: Quantum Computing, St. Olaf College

---

## Table of Contents

1. [Introduction to Quantum Computing Basics](#introduction-to-quantum-computing-basics)
2. [Project Overview](#project-overview)
3. [Part 1: Data Loading and Exploration](#part-1-data-loading-and-exploration)
4. [Part 2: Classical Machine Learning](#part-2-classical-machine-learning)
5. [Part 3: Data Integration](#part-3-data-integration)
6. [Part 4: Quantum Optimization with QAOA](#part-4-quantum-optimization-with-qaoa)
7. [Glossary of Terms](#glossary-of-terms)
8. [Further Reading](#further-reading)

---

## Introduction to Quantum Computing Basics

### Visual Analogy: Classical vs Quantum

**Classical Computer (like a light switch):**
```
State: [ON] or [OFF]  (definite, always one or the other)
```

**Quantum Computer (like a spinning coin):**
```
State: [Heads] + [Tails]  (both at once, until you look)
After measurement: [Heads] OR [Tails]  (collapses to one)
```

### What is a Qubit?

A **qubit** (quantum bit) is the fundamental unit of quantum information, analogous to a classical bit.

**Classical Bit:**
- Can be either 0 or 1
- Like a light switch: ON or OFF

**Quantum Qubit:**
- Can be in a **superposition** of 0 and 1 simultaneously
- Like a spinning coin: it's both heads and tails until you measure it
- When measured, it collapses to either 0 or 1

### Key Quantum Concepts

#### 1. Superposition
A qubit can exist in multiple states at once. Mathematically:
- State = α|0⟩ + β|1⟩
- Where |α|² + |β|² = 1
- |α|² is the probability of measuring 0
- |β|² is the probability of measuring 1

**Example:** A qubit in equal superposition: (1/√2)|0⟩ + (1/√2)|1⟩
- 50% chance of measuring 0
- 50% chance of measuring 1

#### 2. Entanglement
When qubits are entangled, measuring one instantly affects the other, no matter how far apart they are. This is a uniquely quantum phenomenon with no classical equivalent.

#### 3. Quantum Gates
Operations that manipulate qubits:
- **Hadamard (H)**: Creates superposition: |0⟩ → (|0⟩ + |1⟩)/√2
- **Pauli-X (X)**: Bit flip: |0⟩ → |1⟩, |1⟩ → |0⟩
- **Pauli-Z (Z)**: Phase flip: |0⟩ → |0⟩, |1⟩ → -|1⟩
- **CNOT**: Controlled-NOT, creates entanglement between two qubits
- **Rotation gates (RX, RY, RZ)**: Rotate qubit state around axes

#### 4. Measurement
When you measure a qubit, it collapses from superposition to a definite state (0 or 1). You can't measure without collapsing the state.

---

## Project Overview

This project demonstrates quantum computing through three main parts:

1. **Classical ML (KNN/SVM)**: Predict gladiator survival using explainable machine learning
2. **Data Integration**: Connect gladiator data with historical Wikipedia periods
3. **Quantum Optimization (QAOA)**: Find optimal groupings of historical topics using quantum algorithms

---

## Part 1: Data Loading and Exploration

### Code Explanation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

**What this does:**
- `pandas`: Data manipulation and analysis (like Excel in Python)
- `numpy`: Numerical computing (arrays and math operations)
- `matplotlib` & `seaborn`: Creating visualizations and plots

```python
import pennylane as qml
from pennylane import numpy as pnp
```

**What this does:**
- `pennylane`: Quantum computing framework (like TensorFlow for quantum)
- `pennylane.numpy`: Special numpy that works with quantum operations

**Why separate numpy?** PennyLane's numpy tracks gradients for quantum-classical hybrid algorithms.

```python
np.random.seed(777)
pnp.random.seed(777)
```

**What this does:**
- Sets random number generator seeds
- Ensures reproducible results (same random numbers every run)
- Important for debugging and sharing results

### Loading the Gladiator Dataset

```python
gladiator_df = pd.read_csv("gladiator_data.csv")
```

**What this does:**
- Reads CSV file into a pandas DataFrame
- DataFrame = table with rows (gladiators) and columns (features)
- Think of it as a spreadsheet in Python

**The dataset contains:**
- 9,976 gladiators
- 29 features per gladiator (age, height, wins, losses, etc.)
- Target variable: `Survived` (True/False)

### Loading the Wikipedia Dataset

```python
wiki_df = pd.read_csv("wiki_data.csv")
```

**What this does:**
- Loads Wikipedia entries about historical topics
- Contains: title, text, relevans, popularity, ranking
- 1,077 historical topics

---

## Part 2: Classical Machine Learning

### Why Classical ML First?

Before diving into quantum computing, we use classical ML because:
1. **Easier to understand**: No quantum mechanics needed
2. **Explainable**: We can see WHY predictions are made
3. **Fast**: Runs instantly on regular computers
4. **Baseline**: Compare quantum methods against classical ones

### Data Preprocessing

#### Step 1: Feature Selection

```python
numerical_features = ['Age', 'Height', 'Weight', 'Wins', 'Losses', 
                      'Public Favor', 'Mental Resilience', 'Battle Experience']
```

**What this does:**
- Selects numerical features (numbers we can do math with)
- These are continuous values (age = 32, height = 195 cm)

```python
categorical_features = ['Category', 'Special Skills', 'Weapon of Choice', 
                        'Patron Wealth', 'Equipment Quality', 'Health Status']
```

**What this does:**
- Selects categorical features (categories, not numbers)
- Examples: Category = "Thraex", Weapon = "Sica (Curved Sword)"

#### Step 2: Encoding Categorical Variables

```python
for col in categorical_features:
    le = LabelEncoder()
    df_work[col + '_encoded'] = le.fit_transform(df_work[col].astype(str))
```

**What this does:**
- Converts text categories to numbers
- Example: "Low", "Medium", "High" → 0, 1, 2
- **Why?** Machine learning algorithms need numbers, not text

**Example:**
- "Low" → 0
- "Medium" → 1
- "High" → 2

#### Step 3: Normalization

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**What this does:**
- Normalizes features to have mean=0 and standard deviation=1
- **Why?** Different features have different scales:
  - Age: 20-50
  - Height: 150-200 cm
  - Public Favor: 0.0-1.0

**After normalization:**
- All features are on the same scale
- Prevents one feature from dominating others
- Essential for distance-based methods (KNN, SVM)

**Mathematical formula:**
```
normalized_value = (value - mean) / standard_deviation
```

**Example:**
- Original ages: [20, 30, 40, 50]
- Mean: 35, Std: 12.9
- Normalized: [-1.16, -0.39, 0.39, 1.16]
- Now all features are on similar scale!

#### Step 4: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

**What this does:**
- Splits data into training (80%) and testing (20%) sets
- **Training set**: Used to teach the model
- **Test set**: Used to evaluate how well the model learned
- `stratify=y`: Ensures both sets have same proportion of survivors

**Why split?**
- Test on unseen data to check if model generalizes
- Prevents overfitting (memorizing training data)

---

### K-Nearest Neighbors (KNN) - Detailed Explanation

#### What is KNN?

**KNN** is one of the simplest machine learning algorithms. It's like asking your neighbors for advice!

**How it works:**
1. When you want to predict if a gladiator survived:
2. Find the K most similar gladiators in the training set
3. Look at what happened to those K gladiators
4. Predict the majority outcome

**Example:**
- New gladiator: Age=30, Wins=10, Height=180
- Find 5 most similar gladiators (K=5)
- 4 of them survived, 1 died
- Prediction: **Survived** (majority vote)

#### Code Breakdown

```python
k_range = range(3, 21, 2)  # Try k = 3, 5, 7, 9, ..., 19
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_cv, y_train_cv)
    score = knn.score(X_val_cv, y_val_cv)
    k_scores.append(score)
```

**What this does:**
- Tests different values of K (number of neighbors)
- For each K, trains a model and evaluates it
- Finds the best K value

**Why find optimal K?**
- K too small (K=1): Overfitting, sensitive to noise
  - Example: If nearest neighbor is an outlier, prediction is wrong
- K too large (K=100): Underfitting, ignores local patterns
  - Example: Uses too many distant neighbors, loses local structure
- Optimal K: Balances between these extremes
  - Example: K=5-15 often works well

**Visual Example:**
```
K=1:  [X] ← predict based on this one point (risky!)
K=3:  [X][X][X] ← predict based on these 3 (better)
K=5:  [X][X][X][X][X] ← predict based on these 5 (good balance)
K=20: [X][X]...[X] ← too many, loses local patterns
```

**Distance calculation:**
KNN uses **Euclidean distance** to find similar gladiators:

```
distance = √[(age₁ - age₂)² + (height₁ - height₂)² + (wins₁ - wins₂)² + ...]
```

Smaller distance = more similar gladiators

#### Explainability Example

```python
distances, indices = knn_model.kneighbors(sample, n_neighbors=best_k)
```

**What this does:**
- Finds the K nearest neighbors for a test gladiator
- Returns their distances and indices

**Why this is explainable:**
- You can see exactly which gladiators influenced the prediction
- You can check their actual outcomes
- You can verify the prediction makes sense

**Example output:**
```
For gladiator #1234:
  Nearest neighbor 1: Distance 0.45, Survived: Yes
  Nearest neighbor 2: Distance 0.52, Survived: Yes
  Nearest neighbor 3: Distance 0.58, Survived: No
  ...
Prediction: Survived (because 4 out of 5 neighbors survived)
```

---

### Support Vector Machine (SVM) - Detailed Explanation

#### What is SVM?

**SVM** finds the best decision boundary (line/plane) that separates survivors from non-survivors.

**Key concept:**
- Draws a line (or hyperplane in high dimensions) that best separates the two classes
- Maximizes the margin (distance) between the line and the nearest data points
- Those nearest points are called **support vectors**

**Visual analogy:**
Imagine survivors and non-survivors as two groups on a field. SVM draws a line that:
1. Separates the two groups
2. Is as far as possible from both groups (maximum margin)

```
Survivors:     ●  ●  ●
                \   /
                 \ /
Line:  ───────────┼───────────  (maximum margin)
                 / \
                /   \
Non-survivors: ○  ○  ○

Support vectors are the points closest to the line (● and ○)
```

#### Code Breakdown

```python
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
```

**Parameters explained:**

1. **`kernel='rbf'`** (Radial Basis Function):
   - Allows non-linear decision boundaries
   - Can draw curves, not just straight lines
   - **Why?** Real-world data is rarely linearly separable

2. **`C=1.0`** (Regularization parameter):
   - Controls trade-off between margin size and classification errors
   - **C large**: Hard margin, fewer errors, smaller margin
   - **C small**: Soft margin, more errors, larger margin
   - Think: How strict should we be about misclassifications?

3. **`gamma='scale'`**:
   - Controls influence of individual training examples
   - **High gamma**: Each point has small influence (tight boundaries)
   - **Low gamma**: Each point has large influence (smooth boundaries)

#### Support Vectors

```python
print(f"Number of support vectors: {len(svm_model.support_vectors_)}")
```

**What are support vectors?**
- The data points closest to the decision boundary
- These are the "critical" gladiators that define the boundary
- Only these points matter for classification (others can be ignored)

**Why this matters:**
- SVM is memory efficient (only stores support vectors)
- Support vectors are the "hard cases" - gladiators that are hard to classify

---

### Model Evaluation

#### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
```

**What is a confusion matrix?**
A table showing prediction accuracy:

```
                Predicted
              Died  Survived
Actual Died    [a]    [b]
      Survived [c]    [d]
```

- **a**: Correctly predicted deaths (True Negatives)
- **b**: Incorrectly predicted as survived (False Positives)
- **c**: Incorrectly predicted as died (False Negatives)
- **d**: Correctly predicted survivals (True Positives)

**Metrics:**
- **Accuracy**: (a + d) / (a + b + c + d) - Overall correctness
- **Precision**: d / (b + d) - Of predicted survivors, how many actually survived?
- **Recall**: d / (c + d) - Of actual survivors, how many did we catch?

---

## Part 3: Data Integration

### Connecting Gladiators to Historical Periods

```python
def get_historical_period(birth_year):
    if birth_year < 27:
        return "Roman Republic (Before 27 BCE)"
    elif birth_year < 180:
        return "Early Empire (27 BCE - 180 CE)"
    elif birth_year < 284:
        return "Crisis Period (180-284 CE)"
    else:
        return "Late Empire (284+ CE)"
```

**What this does:**
- Maps gladiator birth years to historical periods
- Creates a connection between gladiator data and historical context

**Historical periods:**
1. **Roman Republic**: Before Augustus became emperor
2. **Early Empire**: Pax Romana, stable period
3. **Crisis Period**: Political instability, economic problems
4. **Late Empire**: Reforms, but decline continues

**Why this matters:**
- Survival rates might vary by historical period
- Different eras had different gladiator practices
- Connects individual data to broader historical trends

### Analysis by Period

```python
period_analysis = gladiator_df.groupby('Historical Period').agg({
    'Survived': ['mean', 'count']
})
```

**What this does:**
- Groups gladiators by historical period
- Calculates average survival rate per period
- Counts how many gladiators in each period

**Insights we can discover:**
- Which period had highest/lowest survival rates?
- How many gladiators fought in each period?
- Do survival patterns correlate with historical events?

### Connecting to Wikipedia Data

```python
roman_keywords = ['roman', 'empire', 'republic', 'gladiator', 'ancient', 'history']
relevant_topics = wiki_df[wiki_df['title'].str.lower().str.contains('|'.join(roman_keywords))]
```

**What this does:**
- Searches Wikipedia titles for Roman history keywords
- Finds relevant historical topics
- Creates a bridge between gladiator data and historical knowledge

**Example connections:**
- Gladiator from "Crisis Period" → Wikipedia topic "Roman Empire"
- Survival rate analysis → Historical context from Wikipedia

---

## Part 4: Quantum Optimization with QAOA

### What is QAOA?

**QAOA** (Quantum Approximate Optimization Algorithm) is a quantum algorithm for solving optimization problems.

**Key idea:**
- Uses quantum superposition to explore many solutions simultaneously
- Finds optimal solutions faster than classical methods (for certain problems)
- Hybrid approach: Quantum circuit + Classical optimizer

### The Max-Cut Problem

#### What is Max-Cut?

Given a graph (nodes connected by edges), find a way to split nodes into two groups that maximizes the number of edges between the groups.

**Example:**
```
Graph:  A---B
        |   |
        C---D

Possible cuts:
Cut 1: {A, B} vs {C, D}
  Edges between groups: AB, CD = 2 edges

Cut 2: {A, C} vs {B, D}
  Edges between groups: AB, CD = 2 edges

Cut 3: {A, D} vs {B, C}
  Edges between groups: AB, CD = 2 edges
```

**In our project:**
- Nodes = Historical Wikipedia topics
- Edges = Similarities between topics
- Goal = Find two groups of topics with maximum connections between groups

### Building the Similarity Graph

#### Step 1: Extract Keywords

```python
def extract_keywords(text):
    words = str(text).lower().split()
    keywords = [w for w in words if len(w) > 3]
    return set(keywords)
```

**What this does:**
- Converts text to lowercase
- Splits into words
- Filters out short words (length > 3)
- Returns unique words (set)

**Example:**
- Input: "History of Europe"
- Output: {"history", "europe"}

#### Step 2: Calculate Similarity

```python
# Jaccard similarity
intersection = len(keywords_i & keywords_j)
union = len(keywords_i | keywords_j)
similarity = intersection / union if union > 0 else 0
```

**Jaccard Similarity:**
- Measures how similar two sets are
- Formula: |A ∩ B| / |A ∪ B|
- Range: 0 (no overlap) to 1 (identical)

**Example:**
- Topic 1 keywords: {"history", "europe", "ancient"}
- Topic 2 keywords: {"history", "rome", "ancient"}
- Intersection: {"history", "ancient"} = 2 words
- Union: {"history", "europe", "ancient", "rome"} = 4 words
- Similarity: 2/4 = 0.5

#### Step 3: Create Adjacency Matrix

```python
threshold = 0.1
adjacency_matrix = (similarity_matrix > threshold).astype(int)
```

**What this does:**
- Creates a graph where topics are connected if similarity > 0.1
- Adjacency matrix: 1 = connected, 0 = not connected

**Example adjacency matrix:**
```
     Topic1  Topic2  Topic3
T1     0       1       1
T2     1       0       0
T3     1       0       0
```
- Topic1 connected to Topic2 and Topic3
- Topic2 only connected to Topic1
- Topic3 only connected to Topic1

---

### Visualizing QAOA

**QAOA Flow Diagram:**
```
1. Start: All qubits in superposition
   |+⟩|+⟩|+⟩...|+⟩  (all partitions equally likely)

2. Apply Cost Hamiltonian (gamma)
   → Favors partitions that cut many edges

3. Apply Mixer Hamiltonian (beta)
   → Explores different partitions

4. Repeat steps 2-3 (p times)

5. Measure: Get bitstring representing partition
   [0,1,0,1,...] → Group 0: topics 0,2,... | Group 1: topics 1,3,...
```

### QAOA Implementation - Step by Step

#### Step 1: Cost Hamiltonian

```python
def maxcut_cost_hamiltonian(adjacency_matrix):
    for i in range(n_qubits_opt):
        for j in range(i + 1, n_qubits_opt):
            if adjacency_matrix[i, j] == 1:  # If there's an edge
                obs = qml.PauliZ(i) @ qml.PauliZ(j)
                observables.append(obs)
                coeffs.append(-0.5)
```

**What is a Hamiltonian?**
- In quantum mechanics: Operator representing energy
- In optimization: Represents the cost function we want to minimize
- Each term corresponds to an edge in the graph

**Pauli-Z operators:**
- Z|0⟩ = |0⟩ (eigenvalue +1)
- Z|1⟩ = -|1⟩ (eigenvalue -1)
- Z_i @ Z_j: Product of Z operators on qubits i and j

**Why -0.5 coefficient?**
- Max-Cut cost: (1 - Z_i Z_j) / 2
- We want to minimize, so we use -0.5 * Z_i Z_j
- When Z_i Z_j = -1 (different states), cost is minimized

**Mathematical explanation:**
- If qubits i and j are in same state: Z_i Z_j = +1, cost = -0.5
- If qubits i and j are in different states: Z_i Z_j = -1, cost = +0.5
- We want different states (cut the edge), so we minimize the Hamiltonian

#### Step 2: Mixer Hamiltonian

```python
def mixer_hamiltonian(n_qubits):
    for i in range(n_qubits):
        coeffs.append(1.0)
        observables.append(qml.PauliX(i))
```

**What is the mixer?**
- Allows the algorithm to explore different solutions
- Pauli-X flips qubits: |0⟩ ↔ |1⟩
- Sum of X operators on all qubits

**Why do we need it?**
- Without mixer: Algorithm gets stuck
- With mixer: Can explore different partitions
- Balances exploitation (cost) and exploration (mixer)

#### Step 3: QAOA Circuit

```python
def qaoa_layer(gamma, beta, edges_list):
    # Apply cost Hamiltonian
    for i, j in edges_list:
        qml.CNOT(wires=[i, j])
        qml.RZ(-gamma, wires=j)
        qml.CNOT(wires=[i, j])
    
    # Apply mixer Hamiltonian
    for i in range(n_qubits_opt):
        qml.RX(2 * beta, wires=i)
```

**QAOA Layer Breakdown:**

**Part A: Cost Hamiltonian Evolution**
```python
qml.CNOT(wires=[i, j])
qml.RZ(-gamma, wires=j)
qml.CNOT(wires=[i, j])
```

**What this does:**
- Implements exp(-i*gamma*H_cost)
- CNOT-RZ-CNOT is the decomposition for exp(i*gamma*Z_i*Z_j)
- **CNOT**: Controlled-NOT gate (creates entanglement)
  - If control qubit is |1⟩, flip target qubit
  - If control qubit is |0⟩, do nothing
- **RZ(-gamma)**: Rotate around Z-axis by angle -gamma
- **Why this decomposition?** Direct implementation of Z_i*Z_j requires this gate sequence

**Detailed CNOT-RZ-CNOT Explanation:**

The sequence `CNOT → RZ → CNOT` implements the two-qubit gate exp(i*gamma*Z_i*Z_j).

**Step-by-step:**
1. **First CNOT**: Entangles qubits i and j
   - If qubit i is |1⟩, flip qubit j
   - This creates correlation between the qubits

2. **RZ(-gamma)**: Rotates qubit j around Z-axis
   - The rotation angle depends on the state of qubit i (due to entanglement)
   - This is where the Z_i*Z_j interaction happens

3. **Second CNOT**: Unentangles, but preserves the interaction
   - Reverses the first CNOT
   - The Z_i*Z_j effect is now encoded in the state

**Why this works:**
- Z_i*Z_j measures if qubits are in same state (both |0⟩ or both |1⟩)
- The CNOT gates create a conditional operation
- RZ applies a phase that depends on both qubits' states
- Result: The phase encodes whether qubits are aligned or not

**Mathematical intuition:**
- Z_i*Z_j = +1 when qubits are same (|00⟩ or |11⟩)
- Z_i*Z_j = -1 when qubits are different (|01⟩ or |10⟩)
- The CNOT-RZ-CNOT sequence applies a phase that distinguishes these cases

**Part B: Mixer Hamiltonian Evolution**
```python
qml.RX(2 * beta, wires=i)
```

**What this does:**
- Implements exp(-i*beta*X_i) for each qubit
- **RX(2*beta)**: Rotate around X-axis by angle 2*beta
- Allows qubits to flip between |0⟩ and |1⟩

**Full QAOA Circuit:**
```python
def qaoa_circuit(params, cost_ham, mixer_ham):
    # Initialize in superposition
    for i in range(n_qubits_opt):
        qml.Hadamard(wires=i)
    
    # Apply p layers
    for i in range(p):
        gamma = params[2 * i]
        beta = params[2 * i + 1]
        qaoa_layer(gamma, beta, edges)
    
    # Measure expectation value
    return qml.expval(cost_ham)
```

**Step-by-step:**

1. **Initialization:**
   ```python
   qml.Hadamard(wires=i)
   ```
   - Creates uniform superposition: |+⟩ = (|0⟩ + |1⟩)/√2
   - All possible partitions are equally likely
   - This is quantum parallelism!

2. **Apply QAOA layers:**
   - Each layer has two parameters: gamma (cost) and beta (mixer)
   - More layers (p) = better approximation, but more parameters to optimize
   - Typical: p = 1 to 5 layers

3. **Measurement:**
   ```python
   return qml.expval(cost_ham)
   ```
   - Measures expectation value of cost Hamiltonian
   - Lower value = better solution (we're minimizing)
   - This is what we optimize

#### Step 4: Parameter Optimization

```python
def qaoa_cost(params):
    return qaoa_qnode(params, cost_hamiltonian, mixer_hamiltonian)

opt_qaoa = qml.AdamOptimizer(stepsize=0.1)
for i in range(n_iterations):
    params, cost_val = opt_qaoa.step_and_cost(qaoa_cost, params)
```

**What this does:**
- Uses classical optimizer (Adam) to find best parameters
- **Hybrid approach**: Quantum circuit + Classical optimization
- Iteratively improves parameters to minimize cost

**Optimization process:**
1. Start with random parameters (gamma, beta)
2. Run quantum circuit, get cost
3. Calculate gradient (how to change parameters)
4. Update parameters
5. Repeat until convergence

**Why hybrid?**
- Quantum: Explores solution space efficiently
- Classical: Optimizes parameters (gradient descent)
- Best of both worlds!

#### Step 5: Sampling Solutions

```python
dev_sample = qml.device('default.qubit', wires=n_qubits_opt, shots=n_samples)

@qml.qnode(dev_sample)
def measurement_circuit(params):
    # ... apply QAOA circuit ...
    return [qml.sample(qml.PauliZ(i)) for i in range(n_qubits_opt)]
```

**What this does:**
- Creates device with shots (measurements)
- Samples bitstrings from optimized circuit
- Each sample is a candidate solution

**Why sample?**
- Quantum state is a probability distribution
- We need to extract actual bitstrings (0s and 1s)
- Sample many times, pick the best solution

**Bitstring interpretation:**
- Each qubit = one topic
- 0 = Group 0, 1 = Group 1
- Example: [0, 1, 0, 1, ...] = Topic 0 in Group 0, Topic 1 in Group 1, etc.

**Finding best solution:**
```python
for bitstring in samples:
    cost = calculate_cut_cost(bitstring)
    if cost > best_cost:
        best_cost = cost
        best_bitstring = bitstring
```

**What this does:**
- Evaluates each sampled bitstring
- Calculates how many edges are cut
- Keeps the best one

---

## Glossary of Terms

### Classical Computing Terms

- **Bit**: Binary digit, can be 0 or 1
- **Algorithm**: Step-by-step procedure to solve a problem
- **Machine Learning**: Algorithms that learn patterns from data
- **Supervised Learning**: Learning from labeled examples
- **Classification**: Predicting categories (e.g., survived/died)
- **Feature**: Input variable (e.g., age, height)
- **Label**: Output variable (e.g., survived=True)
- **Training**: Teaching the model using data
- **Testing**: Evaluating model on unseen data
- **Accuracy**: Percentage of correct predictions

### Quantum Computing Terms

- **Qubit**: Quantum bit, can be in superposition
- **Superposition**: Existing in multiple states simultaneously
- **Entanglement**: Quantum correlation between qubits
- **Quantum Gate**: Operation that manipulates qubits
- **Quantum Circuit**: Sequence of quantum gates
- **Measurement**: Collapsing quantum state to classical bit
- **Hamiltonian**: Operator representing energy/cost
- **Expectation Value**: Average value of an observable
- **Variational Algorithm**: Hybrid quantum-classical optimization
- **QAOA**: Quantum Approximate Optimization Algorithm

### Machine Learning Terms

- **KNN**: K-Nearest Neighbors, classification based on similar examples
- **SVM**: Support Vector Machine, finds optimal decision boundary
- **Support Vector**: Data point closest to decision boundary
- **Normalization**: Scaling features to same range
- **Cross-validation**: Testing model on different data splits
- **Overfitting**: Model memorizes training data, fails on new data
- **Underfitting**: Model too simple, can't learn patterns

### Optimization Terms

- **Max-Cut**: Problem of partitioning graph to maximize edges between groups
- **Graph**: Collection of nodes connected by edges
- **Adjacency Matrix**: Matrix showing which nodes are connected
- **Cost Function**: Function to minimize (or maximize)
- **Gradient**: Direction of steepest increase
- **Optimizer**: Algorithm that finds optimal parameters
- **Convergence**: Reaching optimal solution

---

## Further Reading

### Quantum Computing Basics
- **PennyLane Documentation**: https://docs.pennylane.ai/
- **Qiskit Textbook**: https://qiskit.org/textbook/
- **Quantum Computing for the Very Curious**: https://quantum.country/

### Machine Learning
- **Scikit-learn User Guide**: https://scikit-learn.org/stable/user_guide.html
- **Introduction to Statistical Learning**: Free textbook on ML basics

### QAOA
- **Original QAOA Paper**: Farhi, Goldstone, Gutmann (2014)
- **PennyLane QAOA Tutorial**: https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html

### Max-Cut Problem
- **Wikipedia: Maximum Cut**: https://en.wikipedia.org/wiki/Maximum_cut
- **Graph Theory Basics**: Any introductory graph theory textbook

---

## Common Questions

### Q: Why use quantum computing if classical ML works?

**A:** Quantum computing excels at:
- Optimization problems (like Max-Cut)
- Problems with exponential search spaces
- When quantum speedup is possible

For classification, classical ML is often better. Quantum shines in optimization!

### Q: What's the advantage of QAOA over classical optimization?

**A:** 
- **Classical**: Tries solutions one at a time
- **Quantum**: Explores many solutions simultaneously (superposition)
- For large problems, quantum can be exponentially faster

### Q: Can I run this on a real quantum computer?

**A:** Yes! PennyLane supports:
- IBM Quantum
- Google Quantum AI
- IonQ
- Other quantum hardware providers

Just change the device:
```python
dev = qml.device('qiskit.ibmq', wires=n_qubits, backend='ibmq_quito')
```

### Q: How do I choose the number of QAOA layers (p)?

**A:**
- **p=1**: Fast, but may not find optimal solution
- **p=2-3**: Good balance (used in this project)
- **p>5**: Better solutions, but slower and harder to optimize
- Start small, increase if needed

### Q: What if my graph has no edges?

**A:** If similarity threshold is too high, you might get disconnected nodes. Try:
- Lowering the threshold
- Using different similarity metrics
- Preprocessing the data differently

---

## Code Walkthrough: Complete Flow

### 1. Setup and Imports
```python
import pennylane as qml
```
- Loads quantum computing library
- Provides quantum gates, devices, optimizers

### 2. Data Preparation
```python
X_scaled = scaler.fit_transform(X)
```
- Prepares data for machine learning
- Normalizes features for fair comparison

### 3. Classical ML Training
```python
knn_model.fit(X_train, y_train)
```
- Trains KNN on gladiator data
- Learns patterns in survival

### 4. Graph Construction
```python
adjacency_matrix = (similarity_matrix > threshold).astype(int)
```
- Builds graph from Wikipedia topic similarities
- Creates optimization problem

### 5. Quantum Circuit Definition
```python
@qml.qnode(dev_opt)
def qaoa_circuit(params, cost_ham, mixer_ham):
    # ... quantum gates ...
```
- Defines quantum algorithm
- Creates parameterized circuit

### 6. Optimization
```python
params, cost_val = opt_qaoa.step_and_cost(qaoa_cost, params)
```
- Finds optimal parameters
- Minimizes cost function

### 7. Solution Extraction
```python
sample = measurement_circuit(params)
```
- Samples from optimized quantum state
- Extracts classical solution

---

## Tips for Beginners

1. **Start with classical ML**: Understand KNN and SVM before quantum
2. **Visualize everything**: Plot graphs, confusion matrices, results
3. **Experiment with parameters**: Try different K values, thresholds, etc.
4. **Read error messages**: They often tell you exactly what's wrong
5. **Use small examples first**: Test on 5-10 topics before scaling up
6. **Understand the math**: Don't just copy code, understand why it works
7. **Ask questions**: Quantum computing is complex, it's okay to be confused!

---

## Conclusion

This project demonstrates:
- **Classical ML**: Explainable, beginner-friendly machine learning
- **Data Integration**: Connecting different datasets for insights
- **Quantum Optimization**: Using quantum algorithms for hard problems

The key takeaway: **Quantum computing is a tool**, not a replacement for classical methods. Use the right tool for the right job!

**Next Steps:**
1. Experiment with different parameters
2. Try larger graphs (more topics)
3. Compare QAOA with classical optimization
4. Run on real quantum hardware
5. Explore other quantum algorithms

Happy quantum computing! 🚀

