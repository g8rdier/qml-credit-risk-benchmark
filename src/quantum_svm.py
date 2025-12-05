"""
Quantum SVM Module
Implements Quantum Support Vector Machine using Qiskit quantum kernels.

Uses quantum feature maps to create kernel matrices for SVM classification.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pickle
from pathlib import Path
import time
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class QuantumKernelSVM:
    """
    Quantum SVM using quantum feature maps and kernel estimation.

    This class computes quantum kernels using Qiskit and trains a classical
    SVM with the precomputed kernel matrix.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        feature_map_reps: int = 2,
        entanglement: str = 'linear',
        shots: int = 1024,
        random_state: int = 42,
        cache_dir: str = "data/processed"
    ):
        """
        Initialize the Quantum Kernel SVM.

        Args:
            n_qubits: Number of qubits (should match PCA components)
            feature_map_reps: Number of repetitions in feature map circuit
            entanglement: Entanglement pattern ('linear', 'full', 'circular')
            shots: Number of shots for quantum circuit execution
            random_state: Random seed for reproducibility
            cache_dir: Directory to cache kernel matrices
        """
        self.n_qubits = n_qubits
        self.feature_map_reps = feature_map_reps
        self.entanglement = entanglement
        self.shots = shots
        self.random_state = random_state
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize quantum components
        print(f"🔬 Initializing Quantum Kernel SVM:")
        print(f"   - Qubits: {n_qubits}")
        print(f"   - Feature map: ZZFeatureMap (reps={feature_map_reps}, entanglement={entanglement})")
        print(f"   - Shots: {shots}")

        # Create quantum feature map
        self.feature_map = ZZFeatureMap(
            feature_dimension=n_qubits,
            reps=feature_map_reps,
            entanglement=entanglement
        )

        # Create simulator backend
        self.backend = AerSimulator()

        # Create quantum kernel
        self.quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map
        )

        # Classical SVM with precomputed kernel
        self.model: Optional[SVC] = None

        # Cached kernel matrices
        self.K_train: Optional[np.ndarray] = None
        self.K_test: Optional[np.ndarray] = None

        # Timing metrics
        self.kernel_computation_time: Optional[float] = None
        self.training_time: Optional[float] = None
        self.prediction_time: Optional[float] = None

        self.is_trained = False

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for kernel matrix."""
        return self.cache_dir / f"quantum_kernel_{cache_key}.npy"

    def _compute_cache_key(self, X: np.ndarray, prefix: str = "train") -> str:
        """
        Generate cache key based on data shape and quantum parameters.

        Args:
            X: Input data
            prefix: Prefix for cache key (train/test)

        Returns:
            Cache key string
        """
        n_samples = X.shape[0]
        return f"{prefix}_n{n_samples}_q{self.n_qubits}_r{self.feature_map_reps}_{self.entanglement}"

    def compute_kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Compute quantum kernel matrix with caching.

        Args:
            X1: First dataset
            X2: Second dataset (if None, computes K(X1, X1))
            cache_key: Key for caching (if None, auto-generated)
            use_cache: Whether to use cached kernel if available

        Returns:
            Kernel matrix of shape (len(X1), len(X2))
        """
        # Generate cache key if not provided
        if cache_key is None:
            if X2 is None:
                cache_key = self._compute_cache_key(X1, "train")
            else:
                cache_key = f"test_train_n{X1.shape[0]}x{X2.shape[0]}_q{self.n_qubits}"

        cache_path = self._get_cache_path(cache_key)

        # Try to load from cache
        if use_cache and cache_path.exists():
            print(f"📂 Loading cached kernel matrix from: {cache_path.name}")
            K = np.load(cache_path)
            print(f"   ✅ Loaded kernel matrix of shape {K.shape}")
            return K

        # Compute kernel matrix
        print(f"⚛️  Computing quantum kernel matrix...")
        print(f"   - Computing K({X1.shape[0]}, {X2.shape[0] if X2 is not None else X1.shape[0]})")
        print(f"   - This may take several minutes...")

        start_time = time.time()

        if X2 is None:
            # Symmetric kernel matrix
            K = self.quantum_kernel.evaluate(X1)
        else:
            # Asymmetric kernel matrix
            K = self.quantum_kernel.evaluate(X1, X2)

        computation_time = time.time() - start_time

        print(f"   ✅ Kernel computation complete in {computation_time:.2f}s")
        print(f"   - Kernel matrix shape: {K.shape}")

        # Save to cache
        np.save(cache_path, K)
        print(f"   💾 Cached to: {cache_path.name}")

        return K

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'QuantumKernelSVM':
        """
        Train the Quantum SVM.

        Args:
            X_train: Training features (after PCA)
            y_train: Training labels

        Returns:
            Self (for method chaining)
        """
        print(f"\n🎯 Training Quantum Kernel SVM...")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Feature dimensions: {X_train.shape[1]}")
        print(f"   - Qubits: {self.n_qubits}")

        if X_train.shape[1] != self.n_qubits:
            raise ValueError(
                f"Feature dimensions ({X_train.shape[1]}) must match qubits ({self.n_qubits})"
            )

        # Compute training kernel matrix (with caching)
        kernel_start = time.time()
        self.K_train = self.compute_kernel_matrix(X_train, cache_key="train")
        self.kernel_computation_time = time.time() - kernel_start

        # Train classical SVM with precomputed kernel
        print(f"\n🔧 Training SVM with precomputed quantum kernel...")
        train_start = time.time()
        self.model = SVC(kernel='precomputed', random_state=self.random_state)
        self.model.fit(self.K_train, y_train)
        self.training_time = time.time() - train_start

        self.is_trained = True

        print(f"✅ Training complete!")
        print(f"   - Kernel computation: {self.kernel_computation_time:.2f}s")
        print(f"   - SVM optimization: {self.training_time:.2f}s")
        print(f"   - Total time: {self.kernel_computation_time + self.training_time:.2f}s")
        print(f"   - Support vectors: {len(self.model.support_vectors_)}")

        return self

    def predict(self, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.

        Args:
            X_test: Test features
            X_train: Training features (needed to compute test-train kernel)

        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Compute test-train kernel matrix
        cache_key = f"test_train_n{X_test.shape[0]}x{X_train.shape[0]}_q{self.n_qubits}"

        pred_start = time.time()
        K_test_train = self.compute_kernel_matrix(X_test, X_train, cache_key=cache_key)

        predictions = self.model.predict(K_test_train)
        self.prediction_time = time.time() - pred_start

        return predictions

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test set.

        Args:
            X_test: Test features
            y_test: Test labels
            X_train: Training features (needed for kernel computation)

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n📊 Evaluating Quantum SVM...")

        # Get predictions
        y_pred = self.predict(X_test, X_train)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'kernel_computation_time': self.kernel_computation_time,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'total_time': self.kernel_computation_time + self.training_time + self.prediction_time,
            'n_support_vectors': len(self.model.support_vectors_)
        }

        # Print results
        print("\n" + "="*60)
        print("QUANTUM SVM EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"\nTiming:")
        print(f"Kernel computation: {metrics['kernel_computation_time']:.2f}s")
        print(f"SVM training:       {metrics['training_time']:.2f}s")
        print(f"Prediction:         {metrics['prediction_time']:.2f}s")
        print(f"Total:              {metrics['total_time']:.2f}s")
        print(f"\nModel Info:")
        print(f"Support vectors: {metrics['n_support_vectors']}")
        print(f"Qubits: {self.n_qubits}")
        print(f"Feature map reps: {self.feature_map_reps}")
        print("="*60)

        return metrics

    def plot_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray,
        save_path: Optional[str] = "results/confusion_matrix_quantum.png"
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            X_test: Test features
            y_test: Test labels
            X_train: Training features
            save_path: Path to save the plot (None = display only)
        """
        y_pred = self.predict(X_test, X_train)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Bad Credit', 'Good Credit'],
            yticklabels=['Bad Credit', 'Good Credit']
        )
        plt.title(f'Confusion Matrix - Quantum SVM ({self.n_qubits} qubits)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Saved confusion matrix to: {save_path}")

        plt.show()

    def generate_classification_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            X_test: Test features
            y_test: Test labels
            X_train: Training features

        Returns:
            Classification report as string
        """
        y_pred = self.predict(X_test, X_train)
        report = classification_report(
            y_test,
            y_pred,
            target_names=['Bad Credit', 'Good Credit'],
            digits=4
        )

        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT (QUANTUM)")
        print("="*60)
        print(report)

        return report

    def save_model(self, filepath: str = "models/quantum_svm.pkl") -> Path:
        """
        Save trained model.

        Args:
            filepath: Output path for pickle file

        Returns:
            Path to saved file
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_state = {
            'model': self.model,
            'n_qubits': self.n_qubits,
            'feature_map_reps': self.feature_map_reps,
            'entanglement': self.entanglement,
            'kernel_computation_time': self.kernel_computation_time,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_state, f)

        print(f"\n💾 Saved quantum model to: {output_path}")
        return output_path


if __name__ == "__main__":
    # Test the Quantum SVM
    from data_loader import load_credit_data
    from preprocessing import CreditDataPreprocessor

    print("🚀 Testing Quantum SVM Module\n")

    # Load and preprocess data
    X, y = load_credit_data("openml")
    preprocessor = CreditDataPreprocessor(n_components=4)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y)

    # Train and evaluate
    qsvm = QuantumKernelSVM(n_qubits=4, feature_map_reps=2)
    qsvm.train(X_train, y_train)
    metrics = qsvm.evaluate(X_test, y_test, X_train)

    # Generate visualizations
    qsvm.plot_confusion_matrix(X_test, y_test, X_train)
    qsvm.generate_classification_report(X_test, y_test, X_train)

    # Save model
    qsvm.save_model()
