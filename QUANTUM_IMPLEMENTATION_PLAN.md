# Quantum SVM Implementation Plan (Phase 3)

This document outlines the implementation plan for the Quantum SVM component of the BI2 project.

## Overview

**Goal**: Implement a Quantum Support Vector Machine (QSVM) using Qiskit to compare with the classical SVM baseline.

**Core Concept**: Replace the classical kernel function with a quantum kernel computed using quantum circuits.

## Quantum SVM Theory

### Classical SVM Kernel
```
K_classical(x, y) = exp(-γ||x-y||²)  [RBF kernel]
```

### Quantum Kernel
```
K_quantum(x, y) = |⟨φ(x)|φ(y)⟩|²
```
where `φ(x)` is a quantum feature map that encodes classical data into quantum states.

### Advantage
- Quantum feature maps can explore exponentially large Hilbert spaces
- Potential for better expressiveness in certain datasets
- Access to quantum interference and entanglement

## Implementation Steps

### Step 1: Quantum Feature Map Selection

**Options:**

**A. ZZFeatureMap (Recommended for Start)**
```python
from qiskit.circuit.library import ZZFeatureMap

feature_map = ZZFeatureMap(
    feature_dimension=4,  # Number of PCA components
    reps=2,               # Circuit depth
    entanglement='linear' # Qubit connectivity
)
```

**Characteristics:**
- Creates entanglement between qubits
- `U(x) = exp(i * x[i] * Z_i) * exp(i * π * (x[i]-x[j]) * Z_i Z_j)`
- Depth: O(n_features × reps)

**B. ZFeatureMap**
```python
from qiskit.circuit.library import ZFeatureMap

feature_map = ZFeatureMap(
    feature_dimension=4,
    reps=2
)
```

**Characteristics:**
- No entanglement (simpler)
- `U(x) = exp(i * x[i] * Z_i)`
- Faster simulation

**C. PauliFeatureMap**
```python
from qiskit.circuit.library import PauliFeatureMap

feature_map = PauliFeatureMap(
    feature_dimension=4,
    reps=2,
    paulis=['Z', 'ZZ']
)
```

**Characteristics:**
- Most flexible
- Can combine different Pauli operators

### Step 2: Quantum Kernel Estimation

```python
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import Aer

# Select backend
backend = Aer.get_backend('qasm_simulator')

# Create quantum kernel
quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map,
    quantum_instance=backend
)
```

**Process:**
1. Encode data points x and y into quantum states
2. Create circuit to measure overlap: `|⟨φ(x)|φ(y)⟩|²`
3. Execute circuit multiple times (shots)
4. Estimate kernel value from measurements

### Step 3: Integration with Classical SVM

```python
from sklearn.svm import SVC

# Create SVM with quantum kernel
qsvm = SVC(kernel=quantum_kernel.evaluate)
qsvm.fit(X_train, y_train)
```

**How it works:**
- Sklearn SVC accepts custom kernel functions
- `quantum_kernel.evaluate` computes the quantum kernel matrix
- Rest of SVM training is classical

### Step 4: Full Implementation Skeleton

```python
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
import time

class QuantumSVM:
    def __init__(self, n_qubits=4, feature_map_reps=2, shots=1024):
        # Quantum circuit setup
        self.feature_map = ZZFeatureMap(
            feature_dimension=n_qubits,
            reps=feature_map_reps,
            entanglement='linear'
        )

        # Backend
        self.backend = Aer.get_backend('qasm_simulator')

        # Quantum kernel
        self.quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            quantum_instance=self.backend
        )

        # Classical SVM with quantum kernel
        self.model = SVC(kernel=self.quantum_kernel.evaluate)

        self.training_time = None

    def train(self, X_train, y_train):
        print(f"Training Quantum SVM...")
        print(f"  - Qubits: {self.feature_map.num_qubits}")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Computing quantum kernel matrix...")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        print(f"✅ Training complete in {self.training_time:.2f}s")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        # Similar to classical evaluation
        ...
```

## Expected Challenges

### 1. Computational Cost
**Problem**: Quantum simulation is exponentially expensive
- Classical computer simulates quantum states
- Memory: O(2^n_qubits)
- Time: O(2^n_qubits × circuit_depth)

**Impact on BI2 Project**:
- 4 qubits: ~10-100x slower than classical
- 8 qubits: ~100-1000x slower than classical

**Solutions**:
- Use small datasets for initial tests
- Consider cloud quantum hardware (IBM Quantum, AWS Braket)
- Cache kernel matrix to avoid recomputation

### 2. Shot Noise
**Problem**: Quantum measurements are probabilistic
- Each kernel value is estimated from multiple shots
- More shots = better accuracy but slower

**Solutions**:
- Balance shots vs runtime (1024 shots typical)
- Use error mitigation techniques
- Multiple runs for confidence intervals

### 3. Barren Plateaus
**Problem**: Deep circuits may have vanishing gradients
- Makes optimization difficult
- More relevant for VQE/QAOA, less for kernel methods

**Solutions**:
- Keep circuit depth low (reps=2 or 3)
- Use feature maps with known properties

## Minimal Working Example

```python
# quantum_svm_minimal.py
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load preprocessed data
from src.data_loader import load_credit_data
from src.preprocessing import CreditDataPreprocessor

X, y = load_credit_data("openml")
preprocessor = CreditDataPreprocessor(n_components=4)
X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y)

# Quantum feature map
feature_map = ZZFeatureMap(feature_dimension=4, reps=2)

# Quantum kernel
backend = Aer.get_backend('qasm_simulator')
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Train QSVM
print("Training Quantum SVM...")
qsvm = SVC(kernel=quantum_kernel.evaluate)
qsvm.fit(X_train, y_train)

# Evaluate
y_pred = qsvm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Quantum SVM Accuracy: {accuracy:.4f}")
```

## Testing Strategy

### Phase A: Toy Dataset
```python
# Test with tiny dataset first
X_train_tiny = X_train[:20]
y_train_tiny = y_train[:20]
X_test_tiny = X_test[:10]
y_test_tiny = y_test[:10]
```

**Verify:**
- Code runs without errors
- Kernel matrix has correct shape
- Training completes in reasonable time

### Phase B: Feature Map Comparison
Compare different feature maps:
- ZFeatureMap (no entanglement)
- ZZFeatureMap (linear entanglement)
- ZZFeatureMap (full entanglement)

### Phase C: Full Dataset
Run on full train/test split:
- Measure training time
- Compare accuracy with classical
- Analyze support vectors

### Phase D: Ablation Studies
- Effect of `reps` (circuit depth)
- Effect of `shots` (measurement precision)
- Effect of `n_components` (dimensionality)

## Evaluation Metrics

### Performance Metrics
- Accuracy, Precision, Recall, F1 (same as classical)
- Confusion Matrix
- ROC AUC

### Computational Metrics
- **Training Time**: Wall-clock time for kernel matrix + SVM optimization
- **Kernel Computation Time**: Time for quantum circuits only
- **Circuit Depth**: Number of gate layers
- **Total Gates**: Number of quantum operations

### Comparison Metrics
```
┌──────────────────┬──────────┬──────────┬─────────────┐
│ Metric           │ Classical│ Quantum  │ Difference  │
├──────────────────┼──────────┼──────────┼─────────────┤
│ Accuracy         │  0.725   │  0.730   │  +0.005     │
│ Training Time    │  0.15s   │  45.2s   │  +301x      │
│ F1-Score         │  0.712   │  0.718   │  +0.006     │
└──────────────────┴──────────┴──────────┴─────────────┘
```

## Quantum Hardware Options

### Simulators (Free)
- **Qiskit Aer**: Local simulator (up to ~20 qubits)
- **IBM Quantum Simulator**: Cloud-based

### Real Hardware (Requires Account)
- **IBM Quantum**: Free tier available, 5-qubit systems
- **AWS Braket**: Pay-per-shot, various backends
- **IonQ**: Trapped ion quantum computer

**For BI2 Project**: Simulator sufficient for 4-8 qubits

## Expected Results (Hypothesis)

### Accuracy
- **Expected**: Similar to classical (±5%)
- **Reasoning**: Limited qubit count, simulation noise

### Training Time
- **Expected**: 10-100x slower than classical
- **Reasoning**: Quantum simulation overhead

### When Quantum Might Win
- Highly non-linear decision boundaries
- Data that benefits from quantum interference
- Real quantum hardware (no simulation overhead)

## Timeline for Phase 3

**Week 1: Setup & Testing**
- Install Qiskit and dependencies
- Run minimal working example
- Test on toy dataset

**Week 2: Implementation**
- Create `quantum_svm.py` module
- Implement QuantumSVM class
- Add evaluation functions

**Week 3: Experimentation**
- Feature map comparison
- Hyperparameter tuning
- Full dataset training

**Week 4: Analysis & Comparison**
- Generate comparison plots
- Write analysis section
- Prepare presentation

## Code Structure (Planned)

```python
# src/quantum_svm.py

class QuantumSVM:
    def __init__(self, ...):
        # Setup quantum circuits

    def train(self, X_train, y_train):
        # Train with quantum kernel

    def predict(self, X):
        # Make predictions

    def evaluate(self, X_test, y_test):
        # Evaluate performance

    def plot_quantum_kernel_matrix(self):
        # Visualize quantum kernel

    def compare_with_classical(self, classical_svm):
        # Direct comparison

def compare_classical_quantum(X_train, X_test, y_train, y_test):
    # Run both and compare
    ...
```

## Resources

### Documentation
- [Qiskit Textbook - Machine Learning](https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit.html)
- [Qiskit Machine Learning API](https://qiskit.org/documentation/machine-learning/)
- [Quantum Kernels Tutorial](https://qiskit.org/documentation/machine-learning/tutorials/03_quantum_kernel.html)

### Papers
- Havlíček et al. "Supervised learning with quantum-enhanced feature spaces" Nature 567, 209-212 (2019)
- Schuld & Killoran "Quantum Machine Learning in Feature Hilbert Spaces" Phys. Rev. Lett. 122, 040504 (2019)

### Examples
- [Qiskit QSVM Example](https://github.com/Qiskit/qiskit-machine-learning/blob/main/docs/tutorials/03_quantum_kernel.ipynb)

## Next Steps

1. **Install Qiskit**:
   ```bash
   pip install qiskit qiskit-machine-learning
   ```

2. **Test Minimal Example**:
   Create `test_quantum.py` and run minimal working example

3. **Create Module**:
   Implement `src/quantum_svm.py` following the skeleton above

4. **Integrate with Main**:
   Add `--mode quantum` option to `main.py`

5. **Run Comparison**:
   ```bash
   python main.py --mode compare
   ```

---

**Status**: Planning Phase
**Target Completion**: Phase 3 of BI2 Project
**Expected Implementation Time**: 2-4 weeks
