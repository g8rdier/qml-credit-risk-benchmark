# Schnellstart-Anleitung

Starten Sie den QML Kreditrisiko-Benchmark in 5 Minuten mit **pixi** (moderner, schneller Paketmanager).

## Schritt 1: pixi installieren (Einmalige Einrichtung)

```bash
# pixi installieren (schneller, Rust-basierter Paketmanager)
curl -fsSL https://pixi.sh/install.sh | bash

# Shell neu starten oder ausführen:
export PATH="$HOME/.pixi/bin:$PATH"
```

## Schritt 2: Projektabhängigkeiten installieren

```bash
# Alle Abhängigkeiten installieren (erstellt automatisch isolierte Umgebung)
pixi install

# Dies installiert: pandas, numpy, scikit-learn, qiskit, matplotlib, seaborn, etc.
# Viel schneller als pip (verwendet vorkompilierte Binaries von conda-forge)
```

## Schritt 3: Installation überprüfen

```bash
pixi run python verify_pixi.py
```

Dies überprüft:
- Alle erforderlichen Pakete sind installiert
- Projektstruktur ist korrekt
- Module können importiert werden
- Ein schneller End-to-End-Test läuft erfolgreich

## Schritt 4: Klassische SVM-Pipeline ausführen

### Option A: Einfacher Lauf (Standardeinstellungen)

```bash
pixi run run-classical
```

Dies wird:
1. German Credit Daten von OpenML laden
2. Mit 4 PCA-Komponenten vorverarbeiten
3. Eine RBF-Kernel SVM trainieren
4. Evaluieren und Visualisierungen generieren
5. Modelle in `models/` speichern
6. Plots in `results/` speichern

### Option B: Kernel vergleichen

```bash
pixi run compare-kernels
```

Dies vergleicht Linear-, RBF- und Polynomial-Kernel.

### Option C: Verschiedene PCA-Komponenten

```bash
# Für 4-Qubit Quantum-Implementierung
pixi run run-classical-4

# Für 8-Qubit Quantum-Implementierung
pixi run run-classical-8
```

## Schritt 5: Mit Jupyter Notebook erkunden

```bash
pixi run notebook
# Oder Jupyter-Server starten: pixi run notebook-server
```

Dieses Notebook bietet:
- Interaktive Datenexploration
- Schritt-für-Schritt Vorverarbeitungs-Visualisierung
- Hyperparameter-Tuning-Experimente
- Detaillierte Leistungsanalyse

## Erwartete Ausgabe

Nach dem Ausführen von `main.py` sollten Sie sehen:

### Konsolenausgabe:
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

### Generierte Dateien:
```
models/
  ├── preprocessor.pkl            # Angepasster Präprozessor
  └── classical_svm.pkl           # Trainiertes SVM-Modell

results/
  ├── confusion_matrix_classical.png
  └── roc_curve_classical.png
```

## Ergebnisse verstehen

### Wichtige Metriken:

**Genauigkeit (Accuracy)**: Gesamtkorrektheit (Ziel: ~70-75%)
**Präzision**: Wie viele vorhergesagte gute Kredite sind tatsächlich gut
**Recall**: Wie viele tatsächlich gute Kredite erkennen wir
**F1-Wert**: Ausgewogene Metrik, die Präzision und Recall kombiniert
**ROC AUC**: Fähigkeit des Modells, zwischen Klassen zu unterscheiden

### Typische Leistung:
- **Genauigkeit**: 0,70-0,75
- **Trainingszeit**: < 1 Sekunde
- **Support-Vektoren**: ~400-600 (von 800 Trainingsproben)

## Nächste Schritte

1. **Ergebnisse analysieren**: Generierte Plots in `results/` prüfen
2. **Experimentieren**: Verschiedene `--n-components` Werte ausprobieren
3. **Hyperparameter tunen**: Jupyter Notebook für detaillierte Experimente nutzen
4. **Für Quantum vorbereiten**: Nach zufriedenstellender klassischer Baseline zur Quantum-Implementierung übergehen

## Fehlerbehebung

### Problem: "No module named 'sklearn'"
**Lösung**: `pixi install` ausführen

### Problem: "HTTPError: 500 Server Error" (OpenML)
**Lösung**: OpenML könnte temporär nicht erreichbar sein. Die Daten werden nach dem ersten erfolgreichen Download gecacht.

### Problem: Schlechte Leistung (Genauigkeit < 0,60)
**Lösung**:
- Prüfen ob Daten korrekt geladen wurden
- `--compare-kernels` ausprobieren um besten Kernel zu finden
- `--n-components` erhöhen (mehr Features)

### Problem: Sehr lange Trainingszeit
**Lösung**:
- `--n-components` reduzieren
- `kernel='linear'` für schnelleres Training verwenden
- Datensatzgröße prüfen (sollte 1000 Stichproben sein)

## Erweiterte Nutzung

### Einzelne Module verwenden

```python
from src.data_loader import load_credit_data
from src.preprocessing import CreditDataPreprocessor
from src.classical_svm import ClassicalSVM

# Daten laden
X, y = load_credit_data("openml")

# Vorverarbeiten
preprocessor = CreditDataPreprocessor(n_components=4)
X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y)

# Trainieren
svm = ClassicalSVM(kernel='rbf', C=1.0)
svm.train(X_train, y_train)

# Evaluieren
metrics = svm.evaluate(X_test, y_test)
svm.plot_confusion_matrix(X_test, y_test)
```

## Hilfe erhalten

- `README.md` für ausführliche Dokumentation prüfen
- `python main.py --help` für Kommandozeilenoptionen ausführen
- Issue auf GitHub öffnen (falls zutreffend)
- Code-Kommentare in `src/` Modulen durchsehen

---

**Bereit zu starten?**
```bash
pixi run python verify_pixi.py && pixi run run-classical
```

## Warum pixi?

- ⚡ **10-100x schneller** als pip (vorkompilierte Binaries)
- 🔒 **Reproduzierbar** (automatische `pixi.lock` Datei)
- 🎯 **Task-Runner** integriert (kein Makefile nötig)
- 📦 **Besser für Wissenschaft** (conda-forge hat optimierte Pakete)

Siehe `PIXI_GUIDE.md` für vollständige Dokumentation.
