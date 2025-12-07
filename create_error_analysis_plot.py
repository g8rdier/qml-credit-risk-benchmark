#!/usr/bin/env python3
"""
Create Error Analysis Visualization for Thesis
Side-by-side comparison of error patterns and business impact
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_error_analysis_plot():
    """Create comprehensive error analysis visualization."""

    # Data from confusion matrices
    classical_cm = {'TN': 23, 'FP': 37, 'FN': 19, 'TP': 121}
    quantum_cm = {'TN': 3, 'FP': 57, 'FN': 2, 'TP': 138}

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Error Pattern Analysis: Classical vs Quantum SVM\nBI2 Project - Credit Risk Classification',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Error Types Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])

    error_types = ['False\nNegatives\n(FN)', 'False\nPositives\n(FP)']
    classical_errors = [classical_cm['FN'], classical_cm['FP']]
    quantum_errors = [quantum_cm['FN'], quantum_cm['FP']]

    x = np.arange(len(error_types))
    width = 0.35

    bars1 = ax1.bar(x - width/2, classical_errors, width, label='Classical', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, quantum_errors, width, label='Quantum', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Number of Errors', fontweight='bold')
    ax1.set_title('Error Type Comparison', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(error_types)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Add annotations
    ax1.text(0, max(classical_errors[0], quantum_errors[0]) * 1.2,
             f'89% reduction', ha='center', fontsize=10, color='green', fontweight='bold')
    ax1.text(1, max(classical_errors[1], quantum_errors[1]) * 1.1,
             f'54% increase', ha='center', fontsize=10, color='orange', fontweight='bold')

    # 2. Business Cost Impact
    ax2 = fig.add_subplot(gs[0, 1])

    # Cost calculation
    avg_loan = 10000
    default_loss_rate = 0.80
    opportunity_cost = 0.05

    classical_cost_fn = classical_cm['FN'] * avg_loan * default_loss_rate
    classical_cost_fp = classical_cm['FP'] * avg_loan * opportunity_cost
    quantum_cost_fn = quantum_cm['FN'] * avg_loan * default_loss_rate
    quantum_cost_fp = quantum_cm['FP'] * avg_loan * opportunity_cost

    models = ['Classical\nSVM', 'Quantum\nSVM']
    fn_costs = [classical_cost_fn/1000, quantum_cost_fn/1000]  # in thousands
    fp_costs = [classical_cost_fp/1000, quantum_cost_fp/1000]

    x_pos = np.arange(len(models))
    p1 = ax2.bar(x_pos, fn_costs, 0.6, label='Default Losses (FN)', color='#e74c3c', alpha=0.8)
    p2 = ax2.bar(x_pos, fp_costs, 0.6, bottom=fn_costs, label='Lost Opportunities (FP)',
                color='#f39c12', alpha=0.8)

    ax2.set_ylabel('Cost (€ thousands)', fontweight='bold')
    ax2.set_title('Estimated Business Impact', fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add total cost labels
    classical_total = (classical_cost_fn + classical_cost_fp) / 1000
    quantum_total = (quantum_cost_fn + quantum_cost_fp) / 1000

    ax2.text(0, classical_total + 5, f'€{classical_total:.0f}k\ntotal',
             ha='center', fontweight='bold', fontsize=10)
    ax2.text(1, quantum_total + 5, f'€{quantum_total:.0f}k\ntotal',
             ha='center', fontweight='bold', fontsize=10)

    savings_pct = (classical_total - quantum_total) / classical_total * 100
    ax2.text(0.5, max(classical_total, quantum_total) * 1.15,
             f'74% cost reduction', ha='center', fontsize=11,
             color='green', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 3. Error Rates Comparison
    ax3 = fig.add_subplot(gs[0, 2])

    bad_credits = 60
    good_credits = 140

    type_i_classical = classical_cm['FP'] / bad_credits * 100
    type_i_quantum = quantum_cm['FP'] / bad_credits * 100
    type_ii_classical = classical_cm['FN'] / good_credits * 100
    type_ii_quantum = quantum_cm['FN'] / good_credits * 100

    error_rates = ['Type I Error\n(FP Rate)', 'Type II Error\n(FN Rate)']
    classical_rates = [type_i_classical, type_ii_classical]
    quantum_rates = [type_i_quantum, type_ii_quantum]

    x = np.arange(len(error_rates))
    bars1 = ax3.bar(x - width/2, classical_rates, width, label='Classical', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x + width/2, quantum_rates, width, label='Quantum', color='#e74c3c', alpha=0.8)

    ax3.set_ylabel('Error Rate (%)', fontweight='bold')
    ax3.set_title('Error Rates Comparison', fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(error_rates)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 4. Summary Table (Bottom Left)
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.axis('off')

    summary_text = """
    CONFUSION MATRIX BREAKDOWN
    ═══════════════════════════════════════════════════════════════════════════

    Classical SVM:                              Quantum SVM:
    ├─ True Negatives (TN):     23              ├─ True Negatives (TN):      3
    ├─ False Positives (FP):    37              ├─ False Positives (FP):    57
    ├─ False Negatives (FN):    19  ← HIGH RISK ├─ False Negatives (FN):     2  ← LOW RISK ✓
    └─ True Positives (TP):    121              └─ True Positives (TP):    138

    KEY FINDINGS:
    ─────────────────────────────────────────────────────────────────────────

    1. FALSE NEGATIVE REDUCTION (Critical for Credit Risk)
       • Classical: 19 bad credits approved (13.6% Type II error)
       • Quantum: 2 bad credits approved (1.4% Type II error)
       → 89.5% reduction in default risk

    2. TRADE-OFF: Increased False Positives
       • Classical: 37 good credits rejected
       • Quantum: 57 good credits rejected (54% increase)
       → More conservative lending approach

    3. BUSINESS IMPACT (per 200 loan applications)
       • Classical total cost: €170,500 (€152k defaults + €18.5k lost opportunities)
       • Quantum total cost: €44,500 (€16k defaults + €28.5k lost opportunities)
       → €126,000 savings (73.9% cost reduction)

    4. STRATEGIC IMPLICATIONS
       • Quantum SVM: Risk-averse strategy, suitable for conservative lending
       • Classical SVM: Balanced approach, suitable for growth-oriented lending
       • Choice depends on institutional risk appetite and regulatory requirements
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 5. Risk Profile Radar Chart (Bottom Right)
    ax5 = fig.add_subplot(gs[1, 2], projection='polar')

    categories = ['Recall\n(Sensitivity)', 'Precision', 'Specificity',
                  'F1-Score', 'NPV']

    # Calculate metrics
    classical_recall = classical_cm['TP'] / (classical_cm['TP'] + classical_cm['FN'])
    quantum_recall = quantum_cm['TP'] / (quantum_cm['TP'] + quantum_cm['FN'])

    classical_precision = classical_cm['TP'] / (classical_cm['TP'] + classical_cm['FP'])
    quantum_precision = quantum_cm['TP'] / (quantum_cm['TP'] + quantum_cm['FP'])

    classical_specificity = classical_cm['TN'] / (classical_cm['TN'] + classical_cm['FP'])
    quantum_specificity = quantum_cm['TN'] / (quantum_cm['TN'] + quantum_cm['FP'])

    classical_f1 = 2 * (classical_precision * classical_recall) / (classical_precision + classical_recall)
    quantum_f1 = 2 * (quantum_precision * quantum_recall) / (quantum_precision + quantum_recall)

    classical_npv = classical_cm['TN'] / (classical_cm['TN'] + classical_cm['FN'])
    quantum_npv = quantum_cm['TN'] / (quantum_cm['TN'] + quantum_cm['FN'])

    classical_values = [classical_recall, classical_precision, classical_specificity,
                       classical_f1, classical_npv]
    quantum_values = [quantum_recall, quantum_precision, quantum_specificity,
                     quantum_f1, quantum_npv]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    classical_values += classical_values[:1]
    quantum_values += quantum_values[:1]
    angles += angles[:1]

    ax5.plot(angles, classical_values, 'o-', linewidth=2, label='Classical', color='#3498db')
    ax5.fill(angles, classical_values, alpha=0.15, color='#3498db')
    ax5.plot(angles, quantum_values, 'o-', linewidth=2, label='Quantum', color='#e74c3c')
    ax5.fill(angles, quantum_values, alpha=0.15, color='#e74c3c')

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, size=8)
    ax5.set_ylim(0, 1)
    ax5.set_title('Risk Profile Comparison', fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax5.grid(True)

    # Save
    plt.tight_layout()
    save_path = Path('results/error_analysis_comprehensive.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved error analysis visualization to: {save_path}")
    plt.close()


if __name__ == "__main__":
    create_error_analysis_plot()
    print("\n📊 Error analysis visualization created successfully!")
