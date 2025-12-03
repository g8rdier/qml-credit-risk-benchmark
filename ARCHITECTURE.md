# Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QML Credit Risk Benchmark                        │
│                         (BI2 Project)                               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
            ┌───────▼────────┐          ┌──────▼──────┐
            │  Classical SVM │          │ Quantum SVM │
            │  (Completed)   │          │ (Phase 3)   │
            └───────┬────────┘          └──────┬──────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                        ┌─────────▼──────────┐
                        │   Preprocessing    │
                        │  (PCA Critical)    │
                        └─────────┬──────────┘
                                  │
                        ┌─────────▼──────────┐
                        │    Data Loader     │
                        │  (OpenML/CSV)      │
                        └────────────────────┘
```

## Module Breakdown

### 1. Data Layer (`data_loader.py`)

**Responsibilities:**
- Fetch German Credit Data from OpenML
- Load from local CSV files
- Data validation and summary statistics

**Key Classes:**
- `CreditDataLoader`: Main data loading interface

**Outputs:**
- `X`: Feature DataFrame (20+ features)
- `y`: Target Series (binary: 0/1)

**Data Flow:**
```
OpenML API / CSV File
        ↓
CreditDataLoader
        ↓
Raw DataFrame (1000 × 20+)
        ↓
Saved to data/raw/
```

### 2. Preprocessing Layer (`preprocessing.py`)

**Responsibilities:**
- Handle missing values (median/mode imputation)
- Encode categorical variables (one-hot encoding)
- Scale features (StandardScaler)
- **Apply PCA** (dimensionality reduction - CRITICAL for QML)
- Train/test split with stratification

**Key Classes:**
- `CreditDataPreprocessor`: Complete preprocessing pipeline

**Pipeline Stages:**
```
Raw Data (1000 × 20+)
        ↓
Missing Value Handling
        ↓
Categorical Encoding (One-Hot)
        ↓
Train/Test Split (80/20)
        ↓
Feature Scaling (StandardScaler)
        ↓
PCA Reduction (n → 4 default)
        ↓
Final Data (800 × 4 train, 200 × 4 test)
```

**Why PCA is Critical:**
- Quantum simulators limited by qubits
- 1 feature = 1 qubit in quantum feature map
- Must reduce ~60 encoded features → 4-8 features
- Preserves 60-80% variance with 4-8 components

### 3. Classical SVM Layer (`classical_svm.py`)

**Responsibilities:**
- Train classical SVM with various kernels
- Evaluate model performance
- Generate confusion matrix and ROC curves
- Compare different kernels
- Hyperparameter tuning

**Key Classes:**
- `ClassicalSVM`: Main SVM wrapper with evaluation

**Supported Kernels:**
- **Linear**: `K(x, y) = x^T y`
- **RBF** (default): `K(x, y) = exp(-γ||x-y||²)`
- **Polynomial**: `K(x, y) = (γx^T y + r)^d`
- **Sigmoid**: `K(x, y) = tanh(γx^T y + r)`

**Evaluation Metrics:**
- Classification: Accuracy, Precision, Recall, F1-Score
- Ranking: ROC AUC
- Computational: Training Time, Prediction Time
- Model: Number of Support Vectors

### 4. Quantum SVM Layer (`quantum_svm.py`) [FUTURE]

**Planned Responsibilities:**
- Implement quantum feature map
- Quantum kernel estimation
- Integration with classical SVM
- Comparison with classical performance

**Architecture (Planned):**
```
Preprocessed Data (n × 4)
        ↓
Quantum Feature Map (ZZFeatureMap)
        ↓
Quantum Kernel Matrix Estimation
        ↓
Classical SVM with Quantum Kernel
        ↓
Predictions & Evaluation
```

**Quantum Components:**
- **Feature Map**: Encodes classical data into quantum states
- **Kernel Estimation**: Computes kernel matrix using quantum circuits
- **Simulator**: Qiskit's QASM simulator or real quantum hardware

## Data Flow Architecture

```
┌─────────────────┐
│   OpenML API    │
│   Dataset #31   │
└────────┬────────┘
         │ load_from_openml()
         ▼
┌─────────────────┐
│  Raw DataFrame  │
│  1000 × 20+     │
└────────┬────────┘
         │ preprocess_data()
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Encoded Data   │──────│  Transformers:   │
│  1000 × ~60     │      │  - ColumnTransf. │
└────────┬────────┘      │  - StandardScaler│
         │                └──────────────────┘
         │ train_test_split()
         ▼
┌──────────────────┐    ┌──────────────────┐
│  Train: 800 × 60 │    │  Test: 200 × 60  │
└────────┬─────────┘    └────────┬─────────┘
         │                        │
         │ apply_pca()            │ apply_pca()
         ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│  Train: 800 × 4  │    │  Test: 200 × 4   │
└────────┬─────────┘    └────────┬─────────┘
         │                        │
         │ train()                │
         ▼                        │
┌─────────────────┐              │
│  Trained SVM    │              │
│  Model          │              │
└────────┬────────┘              │
         │ predict()              │
         │◄───────────────────────┘
         ▼
┌─────────────────┐
│  Predictions    │
│  & Metrics      │
└─────────────────┘
```

## File Organization

```
qml-credit-risk-benchmark/
│
├── src/                          # Core implementation
│   ├── __init__.py               # Package initialization
│   ├── data_loader.py            # Data loading (OpenML/CSV)
│   ├── preprocessing.py          # Full preprocessing pipeline
│   ├── classical_svm.py          # Classical SVM implementation
│   └── quantum_svm.py            # [FUTURE] Quantum SVM
│
├── data/                         # Data storage
│   ├── raw/                      # Raw data files
│   │   └── german_credit.csv    # Cached dataset
│   └── processed/                # Preprocessed data
│
├── models/                       # Saved models
│   ├── preprocessor.pkl          # Fitted preprocessor
│   ├── classical_svm.pkl         # Trained classical model
│   └── quantum_svm.pkl           # [FUTURE] Trained quantum model
│
├── results/                      # Outputs
│   ├── confusion_matrix_*.png   # Confusion matrices
│   ├── roc_curve_*.png          # ROC curves
│   └── comparison_results.csv   # Comparison table
│
├── notebooks/                    # Jupyter notebooks
│   └── 01_classical_svm_*.ipynb # Interactive exploration
│
├── main.py                       # Main execution script
├── config.py                     # Configuration settings
├── test_installation.py          # Installation verification
├── requirements.txt              # Dependencies
├── README.md                     # Full documentation
├── QUICKSTART.md                 # Quick start guide
└── ARCHITECTURE.md               # This file
```

## Design Principles

### 1. Modularity
Each component is independent and can be used standalone:
```python
# Use just the data loader
from src.data_loader import load_credit_data
X, y = load_credit_data("openml")

# Use just the preprocessor
from src.preprocessing import CreditDataPreprocessor
prep = CreditDataPreprocessor(n_components=4)
```

### 2. Configurability
All parameters are exposed and configurable:
- PCA components (`n_components`)
- Test size (`test_size`)
- SVM hyperparameters (`C`, `gamma`, `kernel`)
- Random state for reproducibility

### 3. Reproducibility
- Fixed random seeds
- Saved preprocessors and models
- Deterministic train/test splits

### 4. Extensibility
Easy to add new components:
- New kernels
- New preprocessing steps
- New evaluation metrics
- Quantum implementation

### 5. Type Safety
All functions use type hints:
```python
def preprocess_data(
    self,
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...
```

## Performance Considerations

### Classical SVM
- **Training Time**: O(n² to n³) where n = number of samples
- **Memory**: O(n²) for kernel matrix
- **Inference**: O(n_sv × n_features) where n_sv = support vectors

**Optimizations:**
- Linear kernel for large datasets
- PCA reduces feature space
- Stratified sampling maintains class balance

### Quantum SVM (Expected)
- **Training Time**: Much longer due to quantum simulation
- **Quantum Circuits**: O(n_qubits × depth)
- **Shots**: Multiple runs needed for quantum measurements

**Trade-offs:**
- Classical: Fast, limited to polynomial kernels
- Quantum: Slow in simulation, access to exponential Hilbert space

## Testing Strategy

### Unit Tests (Per Module)
```python
# Test data loader
python src/data_loader.py

# Test preprocessor
python src/preprocessing.py

# Test classical SVM
python src/classical_svm.py
```

### Integration Test
```python
# Full pipeline test
python test_installation.py
```

### End-to-End Test
```bash
# Complete workflow
python main.py --mode classical
```

## Quantum Implementation Roadmap

### Phase 3A: Basic QSVM
1. Implement quantum feature map (ZZFeatureMap)
2. Quantum kernel estimation
3. Integration with sklearn SVC

### Phase 3B: Optimization
1. Circuit optimization
2. Parameter tuning
3. Hardware testing (if available)

### Phase 4: Comparison
1. Performance metrics comparison
2. Computational cost analysis
3. Scalability study

## References

**Classical SVM:**
- Scikit-learn documentation
- "Pattern Recognition and Machine Learning" by Bishop

**Quantum SVM:**
- Havlíček et al. "Supervised learning with quantum-enhanced feature spaces" (2019)
- Qiskit Machine Learning documentation
- PennyLane tutorials

---

**Version**: 0.1.0 (Classical Implementation Complete)
**Last Updated**: December 2024
