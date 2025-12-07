#!/usr/bin/env python3
"""
Deep Analysis Script for BI2 Thesis
Generates detailed comparison tables and business impact analysis
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def load_models_and_data():
    """Load saved models and preprocessor."""
    print("Loading saved models and preprocessor...")

    with open('models/classical_svm.pkl', 'rb') as f:
        classical_model = pickle.load(f)

    with open('models/quantum_svm.pkl', 'rb') as f:
        quantum_model = pickle.load(f)

    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor_state = pickle.load(f)

    return classical_model, quantum_model, preprocessor_state


def analyze_confusion_matrices():
    """
    Deep dive into confusion matrix patterns.

    Returns detailed breakdown of errors and their business implications.
    """
    print("\n" + "="*80)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*80)

    # From the confusion matrices we can see:
    # Classical: TN=23, FP=37, FN=19, TP=121
    # Quantum: TN=3, FP=57, FN=2, TP=138

    classical_cm = {
        'TN': 23,  # Correctly rejected bad credits
        'FP': 37,  # Incorrectly rejected good credits
        'FN': 19,  # Incorrectly approved bad credits (COSTLY!)
        'TP': 121  # Correctly approved good credits
    }

    quantum_cm = {
        'TN': 3,   # Correctly rejected bad credits
        'FP': 57,  # Incorrectly rejected good credits
        'FN': 2,   # Incorrectly approved bad credits (COSTLY!)
        'TP': 138  # Correctly approved good credits
    }

    total_samples = sum(classical_cm.values())
    bad_credits_actual = classical_cm['TN'] + classical_cm['FP']  # 60
    good_credits_actual = classical_cm['FN'] + classical_cm['TP']  # 140

    print(f"\nTest Set Composition:")
    print(f"  Total samples: {total_samples}")
    print(f"  Bad credits (actual): {bad_credits_actual} ({bad_credits_actual/total_samples*100:.1f}%)")
    print(f"  Good credits (actual): {good_credits_actual} ({good_credits_actual/total_samples*100:.1f}%)")

    # Error Analysis
    print("\n" + "-"*80)
    print("ERROR BREAKDOWN")
    print("-"*80)

    comparison_df = pd.DataFrame({
        'Metric': [
            'True Negatives (TN)',
            'False Positives (FP)',
            'False Negatives (FN)',
            'True Positives (TP)',
            '',
            'Type I Error Rate',
            'Type II Error Rate',
            'Total Errors',
            'Error Rate'
        ],
        'Classical SVM': [
            classical_cm['TN'],
            classical_cm['FP'],
            classical_cm['FN'],
            classical_cm['TP'],
            '',
            f"{classical_cm['FP']/bad_credits_actual*100:.1f}%",
            f"{classical_cm['FN']/good_credits_actual*100:.1f}%",
            classical_cm['FP'] + classical_cm['FN'],
            f"{(classical_cm['FP'] + classical_cm['FN'])/total_samples*100:.1f}%"
        ],
        'Quantum SVM': [
            quantum_cm['TN'],
            quantum_cm['FP'],
            quantum_cm['FN'],
            quantum_cm['TP'],
            '',
            f"{quantum_cm['FP']/bad_credits_actual*100:.1f}%",
            f"{quantum_cm['FN']/good_credits_actual*100:.1f}%",
            quantum_cm['FP'] + quantum_cm['FN'],
            f"{(quantum_cm['FP'] + quantum_cm['FN'])/total_samples*100:.1f}%"
        ],
        'Interpretation': [
            'Bad credits correctly rejected',
            'Good credits wrongly rejected (lost business)',
            'Bad credits wrongly approved (DEFAULT RISK)',
            'Good credits correctly approved',
            '',
            'False Positive Rate (good→bad)',
            'False Negative Rate (bad→good)',
            'Total misclassifications',
            'Overall error rate'
        ]
    })

    print(comparison_df.to_string(index=False))

    # Business Impact Analysis
    print("\n" + "-"*80)
    print("BUSINESS IMPACT ANALYSIS (Financial Perspective)")
    print("-"*80)

    # Typical credit risk costs (example values)
    avg_loan_amount = 10000  # €10,000 average loan
    default_loss_rate = 0.80  # 80% loss on default
    opportunity_cost = 0.05  # 5% profit margin on good loans

    classical_cost = (classical_cm['FN'] * avg_loan_amount * default_loss_rate +
                     classical_cm['FP'] * avg_loan_amount * opportunity_cost)

    quantum_cost = (quantum_cm['FN'] * avg_loan_amount * default_loss_rate +
                   quantum_cm['FP'] * avg_loan_amount * opportunity_cost)

    print(f"\nAssuming:")
    print(f"  - Average loan: €{avg_loan_amount:,}")
    print(f"  - Loss on default: {default_loss_rate*100:.0f}%")
    print(f"  - Opportunity cost (rejected good credit): {opportunity_cost*100:.0f}%")

    print(f"\nEstimated Costs (200 test samples):")
    print(f"  Classical SVM:")
    print(f"    - Default losses (FN): {classical_cm['FN']} × €{avg_loan_amount*default_loss_rate:,.0f} = €{classical_cm['FN']*avg_loan_amount*default_loss_rate:,.0f}")
    print(f"    - Lost opportunities (FP): {classical_cm['FP']} × €{avg_loan_amount*opportunity_cost:,.0f} = €{classical_cm['FP']*avg_loan_amount*opportunity_cost:,.0f}")
    print(f"    - TOTAL COST: €{classical_cost:,.0f}")

    print(f"\n  Quantum SVM:")
    print(f"    - Default losses (FN): {quantum_cm['FN']} × €{avg_loan_amount*default_loss_rate:,.0f} = €{quantum_cm['FN']*avg_loan_amount*default_loss_rate:,.0f}")
    print(f"    - Lost opportunities (FP): {quantum_cm['FP']} × €{avg_loan_amount*opportunity_cost:,.0f} = €{quantum_cm['FP']*avg_loan_amount*opportunity_cost:,.0f}")
    print(f"    - TOTAL COST: €{quantum_cost:,.0f}")

    savings = classical_cost - quantum_cost
    print(f"\n  → QUANTUM SAVINGS: €{savings:,.0f} ({savings/classical_cost*100:.1f}% reduction)")

    # Key Insights
    print("\n" + "-"*80)
    print("KEY INSIGHTS")
    print("-"*80)

    fn_reduction = ((classical_cm['FN'] - quantum_cm['FN']) / classical_cm['FN'] * 100)
    print(f"\n1. FALSE NEGATIVE REDUCTION (Most Critical)")
    print(f"   Classical: {classical_cm['FN']} bad credits approved → potential defaults")
    print(f"   Quantum: {quantum_cm['FN']} bad credits approved → potential defaults")
    print(f"   → {fn_reduction:.1f}% reduction in default risk")

    fp_increase = ((quantum_cm['FP'] - classical_cm['FP']) / classical_cm['FP'] * 100)
    print(f"\n2. FALSE POSITIVE TRADE-OFF")
    print(f"   Classical: {classical_cm['FP']} good credits rejected")
    print(f"   Quantum: {quantum_cm['FP']} good credits rejected")
    print(f"   → {fp_increase:.1f}% increase (more conservative)")

    print(f"\n3. RISK-AVERSE STRATEGY")
    print(f"   Quantum SVM is significantly more risk-averse:")
    print(f"   - Catches {quantum_cm['TP']/good_credits_actual*100:.1f}% of good credits (recall)")
    print(f"   - Only approves {quantum_cm['TN']/(quantum_cm['TN']+quantum_cm['FP'])*100:.1f}% of bad credit applications")
    print(f"   - Suitable for conservative lending policies")

    return comparison_df, classical_cm, quantum_cm


def analyze_pca_variance(preprocessor_state):
    """Analyze PCA variance explained."""
    print("\n" + "="*80)
    print("PCA DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*80)

    pca = preprocessor_state['pca']
    n_components = preprocessor_state['n_components']

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print(f"\nOriginal features: {len(preprocessor_state['feature_names_after_encoding'])}")
    print(f"Reduced to: {n_components} principal components (for {n_components}-qubit QSVM)")

    print(f"\nVariance Explained by Each Component:")
    for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
        print(f"  PC{i+1}: {var*100:5.2f}% (cumulative: {cum_var*100:5.2f}%)")

    print(f"\nTotal variance retained: {cumulative_var[-1]*100:.2f}%")
    print(f"Variance lost: {(1-cumulative_var[-1])*100:.2f}%")

    print("\n" + "-"*80)
    print("CRITICAL OBSERVATION")
    print("-"*80)
    print(f"Despite losing {(1-cumulative_var[-1])*100:.1f}% of data variance through PCA,")
    print(f"both classical and quantum models achieved ~70% accuracy.")
    print(f"\nThis suggests:")
    print(f"  1. The 4 principal components capture the most discriminative information")
    print(f"  2. Additional features may contain noise rather than signal")
    print(f"  3. Dimensionality reduction improved generalization")

    return explained_var, cumulative_var


def create_summary_table():
    """Create final summary table for thesis."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY FOR THESIS")
    print("="*80)

    summary_data = {
        'Dimension': [
            'Classification Performance',
            'Accuracy',
            'Precision',
            'Recall',
            'F1-Score',
            '',
            'Computational Efficiency',
            'Training Time',
            'Prediction Time',
            'Total Runtime',
            '',
            'Error Analysis',
            'False Negatives (FN)',
            'False Positives (FP)',
            'Type II Error Rate',
            '',
            'Business Impact',
            'Risk Profile',
            'Suitable For'
        ],
        'Classical SVM': [
            '',
            '70.00%',
            '75.00%',
            '85.71%',
            '80.00%',
            '',
            '',
            '0.05s',
            '0.003s',
            '0.08s',
            '',
            '',
            '19 (high risk)',
            '37 (moderate)',
            '13.6%',
            '',
            '',
            'Balanced',
            'Standard lending'
        ],
        'Quantum SVM': [
            '',
            '70.50%',
            '70.77%',
            '98.57%',
            '82.39%',
            '',
            '',
            '774.72s',
            '409.40s',
            '1,184.13s',
            '',
            '',
            '2 (very low risk)',
            '57 (higher)',
            '1.4%',
            '',
            '',
            'Risk-averse',
            'Conservative lending'
        ],
        'Winner / Comment': [
            '',
            'Quantum (+0.5%)',
            'Classical (+4.2%)',
            'Quantum (+12.9%)',
            'Quantum (+2.4%)',
            '',
            '',
            'Classical (15,494x faster)',
            'Classical (136,467x faster)',
            'Classical (14,801x faster)',
            '',
            '',
            'Quantum (89% reduction)',
            'Classical (35% fewer)',
            'Quantum (89% reduction)',
            '',
            '',
            'Different strategies',
            'Depends on risk tolerance'
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save to CSV for thesis
    output_path = Path("results/thesis_summary_table.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved summary table to: {output_path}")

    return summary_df


def main():
    """Run complete analysis."""
    print("\n" + "="*80)
    print("BI2 PROJECT - DEEP ANALYSIS FOR THESIS")
    print("Quantum vs Classical SVM Credit Risk Classification")
    print("="*80)

    # Load models
    classical_model, quantum_model, preprocessor_state = load_models_and_data()

    # Run analyses
    comparison_df, classical_cm, quantum_cm = analyze_confusion_matrices()
    explained_var, cumulative_var = analyze_pca_variance(preprocessor_state)
    summary_df = create_summary_table()

    # Save confusion matrix comparison
    cm_comparison_path = Path("results/confusion_matrix_comparison.csv")
    comparison_df.to_csv(cm_comparison_path, index=False)
    print(f"\n✅ Saved confusion matrix analysis to: {cm_comparison_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files for thesis:")
    print("  • results/thesis_summary_table.csv")
    print("  • results/confusion_matrix_comparison.csv")
    print("  • results/comparison_summary.png (already exists)")
    print("  • results/confusion_matrix_classical.png (already exists)")
    print("  • results/confusion_matrix_quantum.png (already exists)")

    print("\n📝 Next Steps for Thesis:")
    print("  1. Use thesis_summary_table.csv for your results section")
    print("  2. Reference the 89% False Negative reduction in discussion")
    print("  3. Discuss business impact: quantum's risk-averse profile")
    print("  4. Explain PCA trade-off: 66% variance loss but maintained accuracy")
    print("  5. Conclude: Quantum shows promise BUT simulation overhead prohibitive")


if __name__ == "__main__":
    main()
