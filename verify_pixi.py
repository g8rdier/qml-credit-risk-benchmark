#!/usr/bin/env python3
"""Quick verification that pixi installed everything correctly."""

print("🧪 Verifying pixi installation...\n")

# Test core packages
try:
    import pandas as pd
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    import qiskit
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("✅ All core packages imported successfully!\n")
    print("Package Versions:")
    print(f"  - pandas: {pd.__version__}")
    print(f"  - numpy: {np.__version__}")
    print(f"  - qiskit: {qiskit.__version__}")
    print(f"  - matplotlib: {plt.matplotlib.__version__}")
    print(f"  - seaborn: {sns.__version__}")

    # Test a simple operation
    print("\n🔬 Testing basic operations...")
    X = np.random.rand(100, 10)
    pca = PCA(n_components=4)
    X_reduced = pca.fit_transform(X)
    print(f"  - PCA: {X.shape} → {X_reduced.shape} ✅")

    svm = SVC()
    print(f"  - SVM initialized: {type(svm).__name__} ✅")

    print("\n✅ pixi environment is fully functional!")
    print("\n🚀 Ready to run your QML benchmark!")

except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
