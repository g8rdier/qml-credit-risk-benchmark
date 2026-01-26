#!/usr/bin/env python3
"""
Regenerate visualizations without re-running simulations.

This script loads the cached models and their metrics, then regenerates
all visualization files. Useful when you want to adjust plot aesthetics
or text without re-computing expensive quantum kernels.

Usage:
    python regenerate_visualizations.py
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import create_comparison_visualization

def load_metrics_from_json():
    """Load saved metrics from JSON file."""

    metrics_path = Path("results/comparison_metrics.json")

    if not metrics_path.exists():
        print(f"❌ Error: {metrics_path} not found")
        print("   Run the comparison first: python main.py --mode compare")
        return None

    with open(metrics_path, 'r') as f:
        data = json.load(f)

    print("✅ Loaded cached metrics from JSON")

    classical_metrics = data['classical']
    quantum_metrics = data['quantum']
    dataset_info = data['dataset']

    print("\n📊 Classical Metrics:")
    print(f"   Accuracy: {classical_metrics['accuracy']:.4f}")
    print(f"   F1-Score: {classical_metrics['f1_score']:.4f}")
    print(f"   Training: {classical_metrics['training_time']:.4f}s")

    print("\n⚛️  Quantum Metrics:")
    print(f"   Accuracy: {quantum_metrics['accuracy']:.4f}")
    print(f"   F1-Score: {quantum_metrics['f1_score']:.4f}")
    print(f"   Training: {quantum_metrics['training_time']:.2f}s")
    print(f"   Kernel computation: {quantum_metrics['kernel_computation_time']:.2f}s")

    return classical_metrics, quantum_metrics, dataset_info


def main():
    """Main execution."""
    print("="*80)
    print("REGENERATING VISUALIZATIONS FROM CACHED METRICS")
    print("="*80)

    # Load metrics from JSON
    result = load_metrics_from_json()
    if result is None:
        sys.exit(1)

    classical_metrics, quantum_metrics, dataset_info = result

    # Get train/test sizes from dataset info
    n_train = dataset_info['n_train']
    n_test = dataset_info['n_test']

    # Create comparison visualization
    print("\n📈 Generating comparison_summary.png...")
    create_comparison_visualization(
        classical_metrics,
        quantum_metrics,
        save_path="results/comparison_summary.png",
        n_train=n_train,
        n_test=n_test
    )

    print("\n✅ Visualization regeneration complete!")
    print("   Updated: results/comparison_summary.png")
    print("\n💡 Note: ROC and Precision-Recall curves require test data and")
    print("   cannot be regenerated from cached metrics alone.")
    print("   Run full comparison to update those: python main.py --mode compare")

if __name__ == "__main__":
    main()
