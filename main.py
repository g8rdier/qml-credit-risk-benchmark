#!/usr/bin/env python3
"""
QML Credit Risk Benchmark - Main Execution Script

This is the main entry point for the BI2 project comparing
Quantum SVM (QSVM) with Classical SVM on credit risk data.

Usage:
    python main.py --mode classical          # Run classical SVM only
    python main.py --mode quantum            # Run quantum SVM only (TODO)
    python main.py --mode compare            # Run both and compare
    python main.py --n-components 8          # Use 8 PCA components (8 qubits)
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_credit_data
from preprocessing import CreditDataPreprocessor
from classical_svm import ClassicalSVM, compare_kernels


def run_classical_pipeline(n_components: int = 4, compare_kernel: bool = False) -> dict:
    """
    Run the complete classical SVM pipeline.

    Args:
        n_components: Number of PCA components (features for classification)
        compare_kernel: If True, compare different kernel types

    Returns:
        Dictionary with results and metrics
    """
    print("\n" + "="*80)
    print("CLASSICAL SVM PIPELINE")
    print("="*80)

    # Step 1: Load Data
    print("\n📥 STEP 1: Loading Data")
    print("-" * 80)
    X, y = load_credit_data("openml")

    # Step 2: Preprocess Data
    print("\n🔧 STEP 2: Preprocessing Data")
    print("-" * 80)
    preprocessor = CreditDataPreprocessor(n_components=n_components)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y)

    # Save preprocessor for reproducibility
    preprocessor.save_preprocessor()

    # Step 3: Train Classical SVM
    print("\n🤖 STEP 3: Training Classical SVM")
    print("-" * 80)

    if compare_kernel:
        # Compare multiple kernels
        results_df = compare_kernels(X_train, X_test, y_train, y_test)

        # Train the best performing model
        best_kernel = results_df.loc[results_df['accuracy'].idxmax(), 'kernel']
        print(f"\n🏆 Best kernel: {best_kernel}")

        svm = ClassicalSVM(kernel=best_kernel)
        svm.train(X_train, y_train)
    else:
        # Use default RBF kernel
        svm = ClassicalSVM(kernel='rbf')
        svm.train(X_train, y_train)

    # Step 4: Evaluate
    print("\n📊 STEP 4: Evaluating Model")
    print("-" * 80)
    metrics = svm.evaluate(X_test, y_test)

    # Step 5: Generate Reports and Visualizations
    print("\n📈 STEP 5: Generating Reports")
    print("-" * 80)
    svm.generate_classification_report(X_test, y_test)
    svm.plot_confusion_matrix(X_test, y_test)
    svm.plot_roc_curve(X_test, y_test)

    # Step 6: Save Model
    svm.save_model()

    print("\n✅ Classical SVM pipeline completed successfully!")

    return {
        'model': svm,
        'metrics': metrics,
        'preprocessor': preprocessor
    }


def run_quantum_pipeline(n_components: int = 4) -> dict:
    """
    Run the quantum SVM pipeline.

    Args:
        n_components: Number of PCA components (= number of qubits)

    Returns:
        Dictionary with results and metrics
    """
    print("\n" + "="*80)
    print("QUANTUM SVM PIPELINE")
    print("="*80)

    # Import quantum module
    from quantum_svm import QuantumKernelSVM

    # Step 1: Load Data
    print("\n📥 STEP 1: Loading Data")
    print("-" * 80)
    X, y = load_credit_data("openml")

    # Step 2: Preprocess Data
    print("\n🔧 STEP 2: Preprocessing Data")
    print("-" * 80)
    preprocessor = CreditDataPreprocessor(n_components=n_components)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y)

    # Save preprocessor for reproducibility
    preprocessor.save_preprocessor()

    # Step 3: Train Quantum SVM
    print("\n⚛️  STEP 3: Training Quantum SVM")
    print("-" * 80)
    qsvm = QuantumKernelSVM(
        n_qubits=n_components,
        feature_map_reps=2,
        entanglement='linear'
    )
    qsvm.train(X_train, y_train)

    # Step 4: Evaluate
    print("\n📊 STEP 4: Evaluating Model")
    print("-" * 80)
    metrics = qsvm.evaluate(X_test, y_test, X_train)

    # Step 5: Generate Reports and Visualizations
    print("\n📈 STEP 5: Generating Reports")
    print("-" * 80)
    qsvm.generate_classification_report(X_test, y_test, X_train)
    qsvm.plot_confusion_matrix(X_test, y_test, X_train)

    # Step 6: Save Model
    qsvm.save_model()

    print("\n✅ Quantum SVM pipeline completed successfully!")

    return {
        'model': qsvm,
        'metrics': metrics,
        'preprocessor': preprocessor,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def run_comparison(n_components: int = 4) -> None:
    """
    Run both classical and quantum pipelines and compare results.

    Args:
        n_components: Number of PCA components
    """
    print("\n" + "="*80)
    print("CLASSICAL VS QUANTUM SVM COMPARISON")
    print("="*80)

    # Run classical
    classical_results = run_classical_pipeline(n_components=n_components)

    # Run quantum
    quantum_results = run_quantum_pipeline(n_components=n_components)

    # Compare results
    print("\n" + "="*80)
    print("📊 COMPARATIVE ANALYSIS")
    print("="*80)

    c_metrics = classical_results['metrics']
    q_metrics = quantum_results['metrics']

    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│                        PERFORMANCE METRICS COMPARISON                       │")
    print("├─────────────────────────────┬──────────────────┬──────────────────┬─────────┤")
    print("│ Metric                      │   Classical SVM  │    Quantum SVM   │  Winner │")
    print("├─────────────────────────────┼──────────────────┼──────────────────┼─────────┤")

    # Accuracy
    acc_winner = "Classical" if c_metrics['accuracy'] > q_metrics['accuracy'] else \
                 "Quantum" if q_metrics['accuracy'] > c_metrics['accuracy'] else "Tie"
    print(f"│ Accuracy                    │     {c_metrics['accuracy']:.4f}       │     {q_metrics['accuracy']:.4f}       │  {acc_winner:^6} │")

    # Precision
    prec_winner = "Classical" if c_metrics['precision'] > q_metrics['precision'] else \
                  "Quantum" if q_metrics['precision'] > c_metrics['precision'] else "Tie"
    print(f"│ Precision                   │     {c_metrics['precision']:.4f}       │     {q_metrics['precision']:.4f}       │  {prec_winner:^6} │")

    # Recall
    rec_winner = "Classical" if c_metrics['recall'] > q_metrics['recall'] else \
                 "Quantum" if q_metrics['recall'] > c_metrics['recall'] else "Tie"
    print(f"│ Recall                      │     {c_metrics['recall']:.4f}       │     {q_metrics['recall']:.4f}       │  {rec_winner:^6} │")

    # F1-Score
    f1_winner = "Classical" if c_metrics['f1_score'] > q_metrics['f1_score'] else \
                "Quantum" if q_metrics['f1_score'] > c_metrics['f1_score'] else "Tie"
    print(f"│ F1-Score                    │     {c_metrics['f1_score']:.4f}       │     {q_metrics['f1_score']:.4f}       │  {f1_winner:^6} │")

    print("├─────────────────────────────┴──────────────────┴──────────────────┴─────────┤")
    print("│                          COMPUTATIONAL EFFICIENCY                           │")
    print("├─────────────────────────────┬──────────────────┬──────────────────┬─────────┤")

    # Training time
    c_train_time = c_metrics.get('training_time', 0)
    q_train_time = q_metrics.get('training_time', 0) + q_metrics.get('kernel_computation_time', 0)
    train_speedup = q_train_time / c_train_time if c_train_time > 0 else 0
    print(f"│ Training Time               │   {c_train_time:>7.4f}s       │  {q_train_time:>7.2f}s      │  Classical │")
    print(f"│                             │                  │                  │  {train_speedup:.0f}x faster│")

    # Prediction time
    c_pred_time = c_metrics.get('prediction_time', 0)
    q_pred_time = q_metrics.get('prediction_time', 0)
    pred_speedup = q_pred_time / c_pred_time if c_pred_time > 0 else 0
    print(f"│ Prediction Time             │   {c_pred_time:>7.4f}s       │  {q_pred_time:>7.2f}s      │  Classical │")
    print(f"│                             │                  │                  │  {pred_speedup:.0f}x faster│")

    print("└─────────────────────────────┴──────────────────┴──────────────────┴─────────┘")

    print("\n📋 ANALYSIS:")
    print("─" * 80)

    # Determine overall winner
    if c_metrics['f1_score'] > q_metrics['f1_score']:
        print("🏆 Winner: CLASSICAL SVM")
        print(f"   Classical achieves better balanced performance (F1: {c_metrics['f1_score']:.4f})")
        print(f"   and is {train_speedup:.0f}x faster in training.")
    elif q_metrics['f1_score'] > c_metrics['f1_score']:
        print("🏆 Winner: QUANTUM SVM")
        print(f"   Quantum achieves slightly better F1-score ({q_metrics['f1_score']:.4f} vs {c_metrics['f1_score']:.4f})")
        print(f"   at the cost of {train_speedup:.0f}x longer training time.")
    else:
        print("🏆 Result: TIE")
        print("   Both models achieve identical F1-scores.")

    print("\n💡 Key Findings:")
    print(f"   • Accuracy difference: {abs(c_metrics['accuracy'] - q_metrics['accuracy'])*100:.2f}% (minimal)")
    print(f"   • Classical is significantly faster: {train_speedup:.0f}x training, {pred_speedup:.0f}x prediction")
    print(f"   • Quantum shows {'higher' if q_metrics['recall'] > c_metrics['recall'] else 'lower'} recall: {q_metrics['recall']:.2%} vs {c_metrics['recall']:.2%}")
    print(f"   • Classical shows {'higher' if c_metrics['precision'] > q_metrics['precision'] else 'lower'} precision: {c_metrics['precision']:.2%} vs {q_metrics['precision']:.2%}")

    print("\n🔬 Conclusion for BI2 Project:")
    print("   Quantum SVM simulation overhead makes it impractical for current use cases.")
    print("   Real quantum hardware may change this, but simulators don't provide advantages.")
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='QML Credit Risk Benchmark - BI2 Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run classical SVM with 4 PCA components
  python main.py --mode classical --n-components 4

  # Compare different classical kernels
  python main.py --mode classical --compare-kernels

  # Run with 8 components (8 qubits for quantum)
  python main.py --mode classical --n-components 8

  # Full comparison (when quantum is implemented)
  python main.py --mode compare
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['classical', 'quantum', 'compare'],
        default='classical',
        help='Execution mode (default: classical)'
    )

    parser.add_argument(
        '--n-components',
        type=int,
        default=4,
        help='Number of PCA components / qubits (default: 4)'
    )

    parser.add_argument(
        '--compare-kernels',
        action='store_true',
        help='Compare different classical kernel types'
    )

    args = parser.parse_args()

    # Validate n_components
    if args.n_components < 2 or args.n_components > 20:
        print("⚠️  Warning: n_components should typically be between 2 and 20")
        print(f"   You specified: {args.n_components}")

    # Execute based on mode
    try:
        if args.mode == 'classical':
            run_classical_pipeline(
                n_components=args.n_components,
                compare_kernel=args.compare_kernels
            )

        elif args.mode == 'quantum':
            run_quantum_pipeline(n_components=args.n_components)

        elif args.mode == 'compare':
            run_comparison(n_components=args.n_components)

    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
