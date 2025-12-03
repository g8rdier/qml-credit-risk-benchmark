# Quick Start Guide

Get up and running with the QML Credit Risk Benchmark in 5 minutes using **pixi** (modern, fast package manager).

## Step 1: Install pixi (One-Time Setup)

```bash
# Install pixi (fast, Rust-based package manager)
curl -fsSL https://pixi.sh/install.sh | bash

# Restart your shell or run:
export PATH="$HOME/.pixi/bin:$PATH"
```

## Step 2: Install Project Dependencies

```bash
# Install all dependencies (creates isolated environment automatically)
pixi install

# This installs: pandas, numpy, scikit-learn, qiskit, matplotlib, seaborn, etc.
# Much faster than pip (uses pre-compiled binaries from conda-forge)
```

## Step 3: Verify Installation

```bash
pixi run python verify_pixi.py
```

This will verify that:
- All required packages are installed
- Project structure is correct
- Modules can be imported
- A quick end-to-end test runs successfully

## Step 4: Run Classical SVM Pipeline

### Option A: Simple Run (Default Settings)

```bash
pixi run run-classical
```

This will:
1. Load German Credit Data from OpenML
2. Preprocess with 4 PCA components
3. Train an RBF kernel SVM
4. Evaluate and generate visualizations
5. Save models to `models/`
6. Save plots to `results/`

### Option B: Compare Kernels

```bash
pixi run compare-kernels
```

This compares Linear, RBF, and Polynomial kernels.

### Option C: Different PCA Components

```bash
# For 4-qubit quantum implementation
pixi run run-classical-4

# For 8-qubit quantum implementation
pixi run run-classical-8
```

## Step 5: Explore with Jupyter Notebook

```bash
pixi run notebook
# Or start Jupyter server: pixi run notebook-server
```

This notebook provides:
- Interactive data exploration
- Step-by-step preprocessing visualization
- Hyperparameter tuning experiments
- Detailed performance analysis

## Expected Output

After running `main.py`, you should see:

### Console Output:
```
================================================================================
CLASSICAL SVM PIPELINE
================================================================================

📥 STEP 1: Loading Data
--------------------------------------------------------------------------------
📥 Loading dataset from OpenML (ID: 31)...
✅ Loaded 1000 samples with 20 features
   Target distribution: {1: 700, 0: 300}

🔧 STEP 2: Preprocessing Data
--------------------------------------------------------------------------------
...

✅ Classical SVM pipeline completed successfully!
```

### Generated Files:
```
models/
  ├── preprocessor.pkl            # Fitted preprocessor
  └── classical_svm.pkl           # Trained SVM model

results/
  ├── confusion_matrix_classical.png
  └── roc_curve_classical.png
```

## Understanding the Results

### Key Metrics to Look For:

**Accuracy**: Overall correctness (target: ~70-75%)
**Precision**: How many predicted good credits are actually good
**Recall**: How many actual good credits we catch
**F1-Score**: Balanced metric combining precision and recall
**ROC AUC**: Model's ability to discriminate between classes

### Typical Performance:
- **Accuracy**: 0.70-0.75
- **Training Time**: < 1 second
- **Support Vectors**: ~400-600 (out of 800 training samples)

## Next Steps

1. **Analyze Results**: Check the generated plots in `results/`
2. **Experiment**: Try different `--n-components` values
3. **Tune Hyperparameters**: Use the Jupyter notebook for detailed experiments
4. **Prepare for Quantum**: Once satisfied with classical baseline, proceed to quantum implementation

## Troubleshooting

### Issue: "No module named 'sklearn'"
**Solution**: Run `pixi install`

### Issue: "HTTPError: 500 Server Error" (OpenML)
**Solution**: OpenML might be temporarily down. The data will be cached after first successful download.

### Issue: Poor performance (accuracy < 0.60)
**Solution**:
- Check if data loaded correctly
- Try `--compare-kernels` to find best kernel
- Increase `--n-components` (more features)

### Issue: Very long training time
**Solution**:
- Reduce `--n-components`
- Use `kernel='linear'` for faster training
- Check your dataset size (should be 1000 samples)

## Advanced Usage

### Using Individual Modules

```python
from src.data_loader import load_credit_data
from src.preprocessing import CreditDataPreprocessor
from src.classical_svm import ClassicalSVM

# Load data
X, y = load_credit_data("openml")

# Preprocess
preprocessor = CreditDataPreprocessor(n_components=4)
X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y)

# Train
svm = ClassicalSVM(kernel='rbf', C=1.0)
svm.train(X_train, y_train)

# Evaluate
metrics = svm.evaluate(X_test, y_test)
svm.plot_confusion_matrix(X_test, y_test)
```

## Getting Help

- Check `README.md` for detailed documentation
- Run `python main.py --help` for command-line options
- Open an issue on GitHub (if applicable)
- Review code comments in `src/` modules

---

**Ready to start?**
```bash
pixi run python verify_pixi.py && pixi run run-classical
```

## Why pixi?

- ⚡ **10-100x faster** than pip (pre-compiled binaries)
- 🔒 **Reproducible** (automatic `pixi.lock` file)
- 🎯 **Task runner** built-in (no Makefile needed)
- 📦 **Better for science** (conda-forge has optimized packages)

See `PIXI_GUIDE.md` for complete documentation.
