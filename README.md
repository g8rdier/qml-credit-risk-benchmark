# Quantum vs Classical SVM Credit Risk Classification: Empirical Benchmark Study

## Academic Context

- **Course:** Business Intelligence 2, 6th Semester
- **Institution:** IU International University of Applied Sciences
- **Supervisor:** Dr. Stefan Nisch
- **Student:** Gregor Kobilarov
- **Dataset:** German Credit Risk Dataset (Kaggle, n=1,000)
- **Primary Contribution:** Production-ready QML benchmark with modern tooling (pixi, kagglehub) comparing quantum and classical SVM performance on structured financial data

## Research Question

> "To what extent can Quantum Machine Learning (QML) approaches, specifically Quantum Support Vector Machines (QSVM), deliver comparable or better classification results on structured financial data than classical methods today?"

## Hypothesis

QSVM achieves similar accuracy in high-dimensional spaces but requires significantly more computational time in simulation due to quantum state simulation overhead.

## Dataset Characteristics

**German Credit Risk Dataset**
- **Source:** Kaggle (kabure/german-credit-data-with-risk)
- **Samples:** 1,000 credit applications
- **Features:** 10 attributes (5 numeric, 5 categorical)
- **Target:** Binary classification (Good Credit: 700, Bad Credit: 300)
- **Task:** Predict creditworthiness based on applicant attributes

## Project Architecture

```
qml-credit-risk-benchmark/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading from OpenML/CSV
│   ├── preprocessing.py        # Cleaning, encoding, scaling, PCA
│   ├── classical_svm.py        # Classical SVM implementation
│   └── quantum_svm.py          # QSVM implementation (TODO)
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Preprocessed data
├── models/                     # Saved models and preprocessors
├── results/                    # Plots and result files
├── notebooks/                  # Jupyter notebooks for exploration
├── main.py                     # Main execution script
├── requirements.txt
└── README.md
```

## Key Features

### Modular Design
- **Data Loader**: Fetches data from OpenML or loads from CSV
- **Preprocessor**: Handles missing values, encoding, scaling, and PCA
- **Classical SVM**: Scikit-learn based with multiple kernel options
- **Quantum SVM**: (In Progress) Qiskit-based quantum kernel

### Critical Pre-processing Pipeline

1. **Missing Value Handling**
   - Numeric: Median imputation
   - Categorical: Mode imputation

2. **Categorical Encoding**
   - One-hot encoding with drop_first=True

3. **Feature Scaling**
   - StandardScaler (critical for SVM performance)

4. **Dimensionality Reduction (PCA)**
   - Reduces features to match available qubits
   - Default: 4 components (4-qubit QSVM)
   - Configurable: 2-20 components

**Why PCA is Critical:**
- Quantum simulators are limited by qubit count
- Each feature requires 1 qubit in quantum feature map
- PCA preserves maximum variance while reducing dimensions

## Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd qml-credit-risk-benchmark

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Run classical SVM with default settings (4 PCA components)
python main.py --mode classical

# Run with 8 PCA components
python main.py --mode classical --n-components 8

# Compare different kernel types
python main.py --mode classical --compare-kernels
```

### Advanced Usage

#### Using Individual Modules

**Data Loading:**
```python
from src.data_loader import load_credit_data

# Load from OpenML
X, y = load_credit_data("openml")

# Load from CSV
X, y = load_credit_data("path/to/data.csv")
```

**Preprocessing:**
```python
from src.preprocessing import CreditDataPreprocessor

preprocessor = CreditDataPreprocessor(n_components=4)
X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y)

# Save preprocessor for later use
preprocessor.save_preprocessor("models/preprocessor.pkl")
```

**Classical SVM:**
```python
from src.classical_svm import ClassicalSVM

# Train model
svm = ClassicalSVM(kernel='rbf', C=1.0)
svm.train(X_train, y_train)

# Evaluate
metrics = svm.evaluate(X_test, y_test)

# Generate visualizations
svm.plot_confusion_matrix(X_test, y_test)
svm.plot_roc_curve(X_test, y_test)

# Save model
svm.save_model("models/classical_svm.pkl")
```

## Evaluation Metrics

The project tracks the following metrics for comparison:

| Metric | Description | Importance |
|--------|-------------|------------|
| **Accuracy** | Overall correctness | Primary metric |
| **Precision** | Positive predictive value | Important for credit risk |
| **Recall** | True positive rate | Critical for catching bad credits |
| **F1-Score** | Harmonic mean of precision/recall | Balanced performance |
| **ROC AUC** | Area under ROC curve | Model discrimination ability |
| **Training Time** | Time to fit model | Computational cost |
| **Prediction Time** | Time for inference | Deployment feasibility |

## Expected Results

### Classical SVM (Baseline)
- **Accuracy**: ~70-75% (typical for this dataset)
- **Training Time**: < 1 second
- **Best Kernel**: RBF (typically)

### Quantum SVM (Hypothesis)
- **Accuracy**: Similar to classical (±5%)
- **Training Time**: Significantly higher (10-100x) due to simulation
- **Advantage**: May improve with real quantum hardware

## Project Timeline

- [x] **Phase 1**: Data loading and preprocessing
- [x] **Phase 2**: Classical SVM benchmark
- [ ] **Phase 3**: Quantum SVM implementation (In Progress)
- [ ] **Phase 4**: Comparative analysis and visualization
- [ ] **Phase 5**: Final report and presentation

## Technical Notes

### PCA Component Selection

| Components | Explained Variance | Use Case |
|------------|-------------------|----------|
| 2 | ~40-50% | Minimal quantum circuit |
| 4 | ~60-70% | Balanced (recommended) |
| 8 | ~80-90% | Maximum information retention |
| 16+ | ~95%+ | Near-original performance |

### Kernel Comparison

**Linear Kernel:**
- Fast, interpretable
- Good for linearly separable data
- Lower computational cost

**RBF Kernel:**
- Most flexible
- Good default choice
- Handles non-linear patterns

**Polynomial Kernel:**
- Captures specific feature interactions
- Can overfit with high degree

**Quantum Kernel:**
- Uses quantum feature map
- Explores exponentially large Hilbert space
- Computationally expensive in simulation

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
**Solution**: Install requirements: `pip install -r requirements.txt`

**Issue**: Memory error during PCA
**Solution**: Reduce `n_components` or use incremental PCA

**Issue**: Poor model performance
**Solution**: Try different kernels with `--compare-kernels` flag

**Issue**: Quantum implementation not working
**Solution**: Ensure Qiskit is installed: `pip install qiskit qiskit-machine-learning`

## Development

### Running Tests
```bash
# Test individual modules
python src/data_loader.py
python src/preprocessing.py
python src/classical_svm.py
```

### Code Style
- Type hints for all function parameters
- Docstrings in Google style
- English comments
- PEP 8 compliant

## Future Work

1. **Quantum SVM Implementation**
   - Implement quantum feature map (ZZFeatureMap, ZFeatureMap)
   - Quantum kernel estimation using Qiskit
   - Integration with classical SVM via custom kernel

2. **Hyperparameter Optimization**
   - Grid search for C and gamma
   - Cross-validation
   - Automated kernel selection

3. **Extended Comparison**
   - Multiple datasets
   - Real quantum hardware (IBM Quantum, IonQ)
   - Comparison with other QML algorithms

4. **Deployment**
   - REST API for model serving
   - Streamlit dashboard for visualization
   - Docker containerization

## References

- German Credit Data: [OpenML](https://www.openml.org/d/31)
- Qiskit Machine Learning: [Documentation](https://qiskit.org/documentation/machine-learning/)
- Scikit-learn SVM: [User Guide](https://scikit-learn.org/stable/modules/svm.html)

## Author

Gregor Kobilarov

## License

This project is for educational purposes as part of a university course.

---

**Status**: Phase 2 Complete (Classical SVM) | Phase 3 In Progress (Quantum SVM)
