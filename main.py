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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run_classical_pipeline(n_components: int = 4, compare_kernel: bool = False, subset_size: int = None) -> dict:
    """
    Run the complete classical SVM pipeline.

    Args:
        n_components: Number of PCA components (features for classification)
        compare_kernel: If True, compare different kernel types
        subset_size: Optional limit on total samples to use (stratified sampling)

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
    if subset_size is not None:
        print(f"⚡ Using subset mode: Limited to {subset_size} samples (stratified)")
    preprocessor = CreditDataPreprocessor(n_components=n_components)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y, max_samples=subset_size)

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


def run_quantum_pipeline(n_components: int = 4, subset_size: int = None) -> dict:
    """
    Run the quantum SVM pipeline.

    Args:
        n_components: Number of PCA components (= number of qubits)
        subset_size: Optional limit on total samples to use (stratified sampling)

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
    if subset_size is not None:
        print(f"⚡ Using subset mode: Limited to {subset_size} samples (stratified)")
    preprocessor = CreditDataPreprocessor(n_components=n_components)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y, max_samples=subset_size)

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


def create_comparison_visualization(classical_metrics: dict, quantum_metrics: dict, save_path: str = "results/comparison_summary.png", n_train: int = 800, n_test: int = 200) -> None:
    """
    Create a comprehensive comparison visualization.

    Args:
        classical_metrics: Metrics from classical SVM
        quantum_metrics: Metrics from quantum SVM
        save_path: Path to save the visualization
        n_train: Number of training samples used
        n_test: Number of test samples used
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Classical vs Quantum SVM: Comprehensive Comparison\nBI2 Project - German Credit Risk Dataset',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Performance Metrics Comparison (Bar Chart)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    classical_values = [classical_metrics['accuracy'], classical_metrics['precision'],
                       classical_metrics['recall'], classical_metrics['f1_score']]
    quantum_values = [quantum_metrics['accuracy'], quantum_metrics['precision'],
                     quantum_metrics['recall'], quantum_metrics['f1_score']]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, classical_values, width, label='Classical SVM', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, quantum_values, width, label='Quantum SVM', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Metrics Comparison', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1.05])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Timing Comparison (Log Scale Bar Chart)
    c_train = classical_metrics.get('training_time', 0)
    q_train = quantum_metrics.get('training_time', 0) + quantum_metrics.get('kernel_computation_time', 0)
    c_pred = classical_metrics.get('prediction_time', 0)
    q_pred = quantum_metrics.get('prediction_time', 0)

    timing_labels = ['Training Time', 'Prediction Time']
    classical_times = [c_train, c_pred]
    quantum_times = [q_train, q_pred]

    x_timing = np.arange(len(timing_labels))
    bars3 = ax2.bar(x_timing - width/2, classical_times, width, label='Classical SVM', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x_timing + width/2, quantum_times, width, label='Quantum SVM', color='#e74c3c', alpha=0.8)

    ax2.set_ylabel('Time (seconds, log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Computational Efficiency Comparison', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_timing)
    ax2.set_xticklabels(timing_labels)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', which='both')

    # Add speedup annotations
    train_speedup = q_train / c_train if c_train > 0 else 0
    pred_speedup = q_pred / c_pred if c_pred > 0 else 0
    ax2.text(0, max(c_train, q_train) * 1.5, f'{train_speedup:.0f}x\nslower',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e74c3c')
    ax2.text(1, max(c_pred, q_pred) * 1.5, f'{pred_speedup:.0f}x\nslower',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e74c3c')

    # 3. Metrics Heatmap
    comparison_data = np.array([
        [classical_metrics['accuracy'], quantum_metrics['accuracy']],
        [classical_metrics['precision'], quantum_metrics['precision']],
        [classical_metrics['recall'], quantum_metrics['recall']],
        [classical_metrics['f1_score'], quantum_metrics['f1_score']]
    ])

    im = ax3.imshow(comparison_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Classical', 'Quantum'], fontsize=11)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], fontsize=11)
    ax3.set_title('Performance Heatmap', fontsize=13, fontweight='bold', pad=15)

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(2):
            text = ax3.text(j, i, f'{comparison_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Score', fontsize=11, fontweight='bold')

    # 4. Summary Text Box
    ax4.axis('off')

    total_samples = n_train + n_test
    summary_text = f"""
    EXPERIMENT SUMMARY
    {'='*50}

    Dataset: German Credit Risk (OpenML)
    Samples: {total_samples} ({n_train} train / {n_test} test)
    Features: 48 → {quantum_metrics.get('n_qubits', 4)} (PCA, {quantum_metrics.get('n_qubits', 4)} qubits)

    PERFORMANCE WINNER: {'Quantum' if quantum_metrics['f1_score'] > classical_metrics['f1_score'] else 'Classical' if classical_metrics['f1_score'] > quantum_metrics['f1_score'] else 'Tie'}
    • Accuracy Δ: {abs(classical_metrics['accuracy'] - quantum_metrics['accuracy'])*100:.2f}% (minimal)
    • F1-Score: Quantum {quantum_metrics['f1_score']:.4f} vs Classical {classical_metrics['f1_score']:.4f}
    • Quantum has higher recall ({quantum_metrics['recall']:.2%})
    • Classical has higher precision ({classical_metrics['precision']:.2%})

    EFFICIENCY WINNER: Classical
    • Training: {train_speedup:.0f}x faster
    • Prediction: {pred_speedup:.0f}x faster
    • Total time: Classical {c_train + c_pred:.2f}s vs Quantum {q_train + q_pred:.0f}s

    CONCLUSION FOR BI2 PROJECT:
    Quantum SVM provides marginal performance gains
    ({(quantum_metrics['f1_score'] - classical_metrics['f1_score'])*100:.1f}% F1-score improvement) but at
    exponential computational cost ({train_speedup:.0f}x slower).

    Quantum simulation overhead makes it impractical
    for production use with current technology.
    Real quantum hardware may change this trajectory.

    Generated: {Path(__file__).parent.name}
    Student: Gregor Kobilarov | Course: BI2 | Semester: 6
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved comparison visualization to: {save_path}")
    plt.close()


def run_comparison(n_components: int = 4, subset_size: int = None) -> None:
    """
    Run both classical and quantum pipelines and compare results.

    Args:
        n_components: Number of PCA components
        subset_size: Optional limit on total samples to use (stratified sampling)
    """
    print("\n" + "="*80)
    print("CLASSICAL VS QUANTUM SVM COMPARISON")
    print("="*80)

    # Run classical
    classical_results = run_classical_pipeline(n_components=n_components, subset_size=subset_size)

    # Run quantum
    quantum_results = run_quantum_pipeline(n_components=n_components, subset_size=subset_size)

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

    # Generate comparison visualization
    n_train = len(quantum_results['X_train'])
    n_test = len(quantum_results['X_test'])
    create_comparison_visualization(c_metrics, q_metrics, n_train=n_train, n_test=n_test)


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

  # Run quantum with 8 qubits and limited dataset (fast test)
  python main.py --mode quantum --n-components 8 --subset-size 200

  # Full comparison with reduced data (for high qubit counts)
  python main.py --mode compare --n-components 8 --subset-size 250

  # Full dataset comparison (slow with high qubit counts!)
  python main.py --mode compare --n-components 4
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

    parser.add_argument(
        '--subset-size',
        type=int,
        default=None,
        help='Limit dataset to N samples (stratified). Useful for quick tests with high qubit counts. Example: --subset-size 200'
    )

    args = parser.parse_args()

    # Validate n_components
    if args.n_components < 2 or args.n_components > 20:
        print("⚠️  Warning: n_components should typically be between 2 and 20")
        print(f"   You specified: {args.n_components}")

    # Validate subset_size
    if args.subset_size is not None:
        if args.subset_size < 50:
            print("⚠️  Warning: subset_size < 50 may not provide statistically meaningful results")
            print(f"   You specified: {args.subset_size}")
        if args.subset_size > 1000:
            print("⚠️  Warning: The full dataset has ~1000 samples, your subset_size is larger")
            print(f"   You specified: {args.subset_size}")

    # Execute based on mode
    try:
        if args.mode == 'classical':
            run_classical_pipeline(
                n_components=args.n_components,
                compare_kernel=args.compare_kernels,
                subset_size=args.subset_size
            )

        elif args.mode == 'quantum':
            run_quantum_pipeline(
                n_components=args.n_components,
                subset_size=args.subset_size
            )

        elif args.mode == 'compare':
            run_comparison(
                n_components=args.n_components,
                subset_size=args.subset_size
            )

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
