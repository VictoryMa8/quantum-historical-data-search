# Quick Reference: Quantum Computing Project

## Code Structure Overview

```
main.ipynb
├── Cell 0: Imports and Setup
├── Cell 1-3: Data Loading
├── Cell 4-9: Classical ML (KNN/SVM)
├── Cell 10-16: Quantum Optimization (QAOA)
└── Cell 17: Summary
```

## Key Code Snippets Explained

### 1. Quantum Device Creation
```python
dev = qml.device('default.qubit', wires=n_qubits)
```
- Creates a quantum simulator
- `wires=n_qubits`: Number of qubits
- `default.qubit`: PennyLane's built-in simulator

### 2. Quantum Circuit Decorator
```python
@qml.qnode(dev)
def quantum_circuit(params):
    # quantum gates here
    return qml.expval(qml.PauliZ(0))
```
- `@qml.qnode`: Marks function as quantum circuit
- `qml.expval`: Returns expectation value (not a sample)

### 3. Quantum Gates Cheat Sheet

| Gate | Symbol | Effect | Code |
|------|--------|--------|------|
| Hadamard | H | Creates superposition | `qml.Hadamard(wires=i)` |
| Pauli-X | X | Bit flip (NOT) | `qml.PauliX(wires=i)` |
| Pauli-Z | Z | Phase flip | `qml.PauliZ(wires=i)` |
| CNOT | CX | Controlled-NOT | `qml.CNOT(wires=[control, target])` |
| Rotation-X | RX | Rotate around X-axis | `qml.RX(angle, wires=i)` |
| Rotation-Y | RY | Rotate around Y-axis | `qml.RY(angle, wires=i)` |
| Rotation-Z | RZ | Rotate around Z-axis | `qml.RZ(angle, wires=i)` |
| Rot | R | General rotation | `qml.Rot(phi, theta, omega, wires=i)` |

### 4. Measurement Types

```python
# Expectation value (for optimization)
qml.expval(qml.PauliZ(0))  # Returns: -1.0 to 1.0

# Sample (requires shots)
qml.sample(qml.PauliZ(0))  # Returns: -1 or 1 (random)
```

**Key difference:**
- `expval`: Average value (deterministic, no shots needed)
- `sample`: Random measurement (requires shots parameter)

### 5. QAOA Circuit Pattern

```python
# 1. Initialize superposition
for i in range(n_qubits):
    qml.Hadamard(wires=i)

# 2. Apply cost Hamiltonian (for each edge)
for i, j in edges:
    qml.CNOT(wires=[i, j])
    qml.RZ(-gamma, wires=j)
    qml.CNOT(wires=[i, j])

# 3. Apply mixer Hamiltonian
for i in range(n_qubits):
    qml.RX(2 * beta, wires=i)

# 4. Measure
return qml.expval(cost_hamiltonian)
```

### 6. Common Patterns

#### Pattern 1: Creating Superposition
```python
qml.Hadamard(wires=0)  # |0⟩ → (|0⟩ + |1⟩)/√2
```

#### Pattern 2: Creating Entanglement
```python
qml.Hadamard(wires=0)   # Create superposition
qml.CNOT(wires=[0, 1])  # Entangle qubits 0 and 1
# Result: (|00⟩ + |11⟩)/√2 (Bell state)
```

#### Pattern 3: Encoding Classical Data
```python
# Angle encoding
qml.RY(feature_value, wires=i)  # Rotate by feature value
```

#### Pattern 4: Parameterized Rotation
```python
qml.Rot(params[0], params[1], params[2], wires=i)
# params are trainable (requires_grad=True)
```

## Common Errors and Fixes

### Error: "Measurement sample not accepted for analytic simulation"
**Fix:** Add `shots` parameter to device:
```python
dev = qml.device('default.qubit', wires=n_qubits, shots=1000)
```

### Error: "Module not found"
**Fix:** Install in correct environment:
```bash
conda activate quantum_computing
pip install package_name
```

### Error: "Gradient computation failed"
**Fix:** Ensure parameters have `requires_grad=True`:
```python
params = pnp.array([...], requires_grad=True)
```

## Performance Tips

1. **Use smaller datasets for testing**: Start with 10-20 topics
2. **Reduce QAOA layers**: p=1 or p=2 for faster execution
3. **Use fewer shots**: 100-500 instead of 1000 for testing
4. **Batch operations**: Process multiple samples at once when possible

## Testing Checklist

- [ ] All imports work
- [ ] Data loads correctly
- [ ] Classical ML trains successfully
- [ ] Quantum circuit runs without errors
- [ ] Optimization converges
- [ ] Sampling produces valid results
- [ ] Visualizations display correctly

## Next Steps After Running

1. Experiment with different K values (KNN)
2. Try different SVM kernels (linear, polynomial)
3. Adjust similarity threshold for graph
4. Increase QAOA layers (p)
5. Try different optimizers (GradientDescent, NesterovMomentum)
6. Visualize quantum circuit: `qml.draw(qaoa_circuit)(params, cost_ham, mixer_ham)`

