"""
Classical SVM Module
Implements classical Support Vector Machine as the benchmark model.

Uses scikit-learn's SVC with various kernels (linear, RBF, polynomial).
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
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
from pathlib import Path
import pickle
import time


class ClassicalSVM:
    """
    Classical SVM classifier with comprehensive evaluation.

    This serves as the baseline for comparison with QSVM.
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        random_state: int = 42
    ):
        """
        Initialize the classical SVM.

        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
            random_state: Random seed for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state

        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True  # Enable probability estimates for ROC
        )

        self.is_trained = False
        self.training_time: Optional[float] = None
        self.prediction_time: Optional[float] = None

        print(f"🤖 Initialized Classical SVM:")
        print(f"   - Kernel: {kernel}")
        print(f"   - C: {C}")
        print(f"   - Gamma: {gamma}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'ClassicalSVM':
        """
        Train the SVM model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Self (for method chaining)
        """
        print(f"\n🎯 Training Classical SVM ({self.kernel} kernel)...")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Feature dimensions: {X_train.shape[1]}")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        self.is_trained = True

        print(f"✅ Training complete in {self.training_time:.4f} seconds")
        print(f"   - Support vectors: {len(self.model.support_vectors_)}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        start_time = time.time()
        predictions = self.model.predict(X)
        self.prediction_time = time.time() - start_time

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n📊 Evaluating Classical SVM...")

        # Get predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]  # Probability of positive class

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'n_support_vectors': len(self.model.support_vectors_)
        }

        # Print results
        print("\n" + "="*60)
        print("CLASSICAL SVM EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"\nTiming:")
        print(f"Training time:   {metrics['training_time']:.4f}s")
        print(f"Prediction time: {metrics['prediction_time']:.4f}s")
        print(f"\nModel Info:")
        print(f"Support vectors: {metrics['n_support_vectors']}")
        print("="*60)

        return metrics

    def plot_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[str] = "results/confusion_matrix_classical.png"
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot (None = display only)
        """
        y_pred = self.predict(X_test)
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
        plt.title(f'Confusion Matrix - Classical SVM ({self.kernel} kernel)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Saved confusion matrix to: {save_path}")

        plt.show()

    def plot_roc_curve(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[str] = "results/roc_curve_classical.png"
    ) -> None:
        """
        Plot ROC curve.

        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot (None = display only)
        """
        y_proba = self.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'Classical SVM (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Classical SVM ({self.kernel} kernel)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Saved ROC curve to: {save_path}")

        plt.show()

    def generate_classification_report(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """
        Generate detailed classification report.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Classification report as string
        """
        y_pred = self.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=['Bad Credit', 'Good Credit'],
            digits=4
        )

        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        print(report)

        return report

    def save_model(self, filepath: str = "models/classical_svm.pkl") -> Path:
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
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_state, f)

        print(f"\n💾 Saved model to: {output_path}")
        return output_path

    @classmethod
    def load_model(cls, filepath: str = "models/classical_svm.pkl") -> 'ClassicalSVM':
        """
        Load a saved model.

        Args:
            filepath: Path to pickle file

        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)

        # Create instance
        svm = cls(
            kernel=model_state['kernel'],
            C=model_state['C'],
            gamma=model_state['gamma']
        )

        # Restore state
        svm.model = model_state['model']
        svm.is_trained = True
        svm.training_time = model_state['training_time']
        svm.prediction_time = model_state['prediction_time']

        print(f"📂 Loaded model from: {filepath}")
        return svm


def compare_kernels(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    kernels: list = ['linear', 'rbf', 'poly']
) -> pd.DataFrame:
    """
    Compare performance of different SVM kernels.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        kernels: List of kernel types to compare

    Returns:
        DataFrame with comparison results
    """
    print("\n" + "="*60)
    print("KERNEL COMPARISON")
    print("="*60)

    results = []

    for kernel in kernels:
        print(f"\n🔬 Testing {kernel} kernel...")

        svm = ClassicalSVM(kernel=kernel)
        svm.train(X_train, y_train)
        metrics = svm.evaluate(X_test, y_test)
        metrics['kernel'] = kernel

        results.append(metrics)

    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results[['kernel', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'training_time']]

    print("\n" + "="*60)
    print("KERNEL COMPARISON SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)

    return df_results


if __name__ == "__main__":
    # Test the Classical SVM
    from data_loader import load_credit_data
    from preprocessing import CreditDataPreprocessor

    print("🚀 Testing Classical SVM Module\n")

    # Load and preprocess data
    X, y = load_credit_data("openml")
    preprocessor = CreditDataPreprocessor(n_components=4)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y)

    # Train and evaluate single model
    svm = ClassicalSVM(kernel='rbf')
    svm.train(X_train, y_train)
    metrics = svm.evaluate(X_test, y_test)

    # Generate visualizations
    svm.plot_confusion_matrix(X_test, y_test)
    svm.plot_roc_curve(X_test, y_test)
    svm.generate_classification_report(X_test, y_test)

    # Save model
    svm.save_model()

    # Compare different kernels
    print("\n" + "="*60)
    compare_kernels(X_train, X_test, y_train, y_test)
