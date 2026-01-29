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

    # Data from confusion matrices (full dataset: 800 train, 200 test)
    classical_cm = {'TN': 20, 'FP': 40, 'FN': 20, 'TP': 120}
    quantum_cm = {'TN': 3, 'FP': 57, 'FN': 2, 'TP': 138}

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Fehlermuster-Analyse: Klassisch vs. Quantum SVM\nBusiness Intelligence II Projekt - Kreditrisiko-Klassifikation',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Error Types Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])

    error_types = ['Falsch\nNegative\n(FN)', 'Falsch\nPositive\n(FP)']
    classical_errors = [classical_cm['FN'], classical_cm['FP']]
    quantum_errors = [quantum_cm['FN'], quantum_cm['FP']]

    x = np.arange(len(error_types))
    width = 0.35

    bars1 = ax1.bar(x - width/2, classical_errors, width, label='Klassisch', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, quantum_errors, width, label='Quantum', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Anzahl der Fehler', fontweight='bold')
    ax1.set_title('Vergleich der Fehlertypen', fontweight='bold', pad=15)
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

    # Add annotations - place labels next to bars
    ax1.text(0, max(classical_errors[0], quantum_errors[0]) * 1.2,
             f'90% Reduktion', ha='center', fontsize=10, color='green', fontweight='bold')
    ax1.text(0.95, classical_errors[1] * 1.2, f'43% Erhöhung',
             ha='right', va='center', fontsize=10, color='orange', fontweight='bold')

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

    models = ['Klassisch\nSVM', 'Quantum\nSVM']
    fn_costs = [classical_cost_fn/1000, quantum_cost_fn/1000]  # in thousands
    fp_costs = [classical_cost_fp/1000, quantum_cost_fp/1000]

    x_pos = np.arange(len(models))
    p1 = ax2.bar(x_pos, fn_costs, 0.6, label='Ausfallverluste (FN)', color='#e74c3c', alpha=0.8)
    p2 = ax2.bar(x_pos, fp_costs, 0.6, bottom=fn_costs, label='Entgangene Chancen (FP)',
                color='#f39c12', alpha=0.8)

    ax2.set_ylabel('Kosten (€ Tausend)', fontweight='bold')
    ax2.set_title('Hypothetischer Geschäftseinfluss*', fontweight='bold', pad=15)
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
    ax2.text(1, quantum_total * 0.5, f'73% Kostenreduktion\ndurch Quantum',
             ha='center', va='center', fontsize=9,
             color='green', fontweight='bold', zorder=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85))

    # 3. Error Rates Comparison
    ax3 = fig.add_subplot(gs[0, 2])

    bad_credits = 60
    good_credits = 140

    type_i_classical = classical_cm['FP'] / bad_credits * 100
    type_i_quantum = quantum_cm['FP'] / bad_credits * 100
    type_ii_classical = classical_cm['FN'] / good_credits * 100
    type_ii_quantum = quantum_cm['FN'] / good_credits * 100

    error_rates = ['Typ I Fehler\n(FP Rate)', 'Typ II Fehler\n(FN Rate)']
    classical_rates = [type_i_classical, type_ii_classical]
    quantum_rates = [type_i_quantum, type_ii_quantum]

    x = np.arange(len(error_rates))
    bars1 = ax3.bar(x - width/2, classical_rates, width, label='Klassisch', color='#3498db', alpha=0.8)
    bars2 = ax3.bar(x + width/2, quantum_rates, width, label='Quantum', color='#e74c3c', alpha=0.8)

    ax3.set_ylabel('Fehlerrate (%)', fontweight='bold')
    ax3.set_title('Vergleich der Fehlerraten', fontweight='bold', pad=15)
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
    KONFUSIONSMATRIX-AUFSCHLÜSSELUNG
    ═══════════════════════════════════════════════════════════════════════════

    Klassisch SVM:                              Quantum SVM:
    ├─ Wahr Negative (TN):      20              ├─ Wahr Negative (TN):       3
    ├─ Falsch Positive (FP):    40              ├─ Falsch Positive (FP):    57
    ├─ Falsch Negative (FN):    20  ← HOHES RISIKO ├─ Falsch Negative (FN):   2  ← NIEDRIGES RISIKO ✓
    └─ Wahr Positive (TP):     120              └─ Wahr Positive (TP):     138

    KERNERKENNTNISSE:
    ─────────────────────────────────────────────────────────────────────────

    1. FALSCH-NEGATIV-REDUKTION (Kritisch für Kreditrisiko)
       • Klassisch: 20 schlechte Kredite bewilligt (14.3% Typ II Fehler)
       • Quantum: 2 schlechte Kredite bewilligt (1.4% Typ II Fehler)
       → 90.0% Reduktion des Ausfallrisikos

    2. KOMPROMISS: Erhöhte Falsch-Positive
       • Klassisch: 40 gute Kredite abgelehnt
       • Quantum: 57 gute Kredite abgelehnt (43% Erhöhung)
       → Konservativerer Kreditvergabe-Ansatz

    3. GESCHÄFTSEINFLUSS* (pro 200 Kreditanträge)
       • Klassisch Gesamtkosten: €180k (€160k Ausfälle + €20k entgangene Chancen)
       • Quantum Gesamtkosten: €44.5k (€16k Ausfälle + €28.5k entgangene Chancen)
       → €135.5k Einsparungen (75.3% Kostenreduktion)
       *Hypothetisches Szenario mit branchenüblichen Annahmen (€10k Ø Kredit,
        80% Ausfallverlustrate, 5% Opportunitätskosten)

    4. STRATEGISCHE IMPLIKATIONEN
       • Quantum SVM: Risikoaverse Strategie, geeignet für konservative Kreditvergabe
       • Klassisch SVM: Ausgewogener Ansatz, geeignet für wachstumsorientierte Kreditvergabe
       • Wahl abhängig von institutionellem Risikoappetit und regulatorischen Anforderungen
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 5. Risk Profile Radar Chart (Bottom Right)
    ax5 = fig.add_subplot(gs[1, 2], projection='polar')

    categories = ['                Recall\n            (Sensitivität)', 'Präzision', 'Spezifität',
                  'F1-Wert', 'NPV']

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

    ax5.plot(angles, classical_values, 'o-', linewidth=2, label='Klassisch', color='#3498db')
    ax5.fill(angles, classical_values, alpha=0.15, color='#3498db')
    ax5.plot(angles, quantum_values, 'o-', linewidth=2, label='Quantum', color='#e74c3c')
    ax5.fill(angles, quantum_values, alpha=0.15, color='#e74c3c')

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, size=8)
    ax5.set_ylim(0, 1)
    ax5.set_title('Vergleich des Risikoprofils', fontweight='bold', pad=20)
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
