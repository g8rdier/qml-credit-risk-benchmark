# Projektarchitektur

## Systemübersicht

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QML Kreditrisiko-Benchmark                       │
│                   (Business Intelligence II-Projekt)                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
            ┌───────▼────────┐          ┌──────▼──────┐
            │ Klassische SVM │          │ Quantum SVM │
            │  (Abgeschlossen)│          │ (Phase 3)   │
            └───────┬────────┘          └──────┬──────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                        ┌─────────▼──────────┐
                        │   Vorverarbeitung  │
                        │  (PCA Kritisch)    │
                        └─────────┬──────────┘
                                  │
                        ┌─────────▼──────────┐
                        │    Data Loader     │
                        │  (OpenML/CSV)      │
                        └────────────────────┘
```

## Modulübersicht

### 1. Datenschicht (`data_loader.py`)

**Verantwortlichkeiten:**
- German Credit Daten von OpenML abrufen
- Aus lokalen CSV-Dateien laden
- Datenvalidierung und zusammenfassende Statistiken

**Wichtige Klassen:**
- `CreditDataLoader`: Haupt-Datenlade-Schnittstelle

**Ausgaben:**
- `X`: Feature DataFrame (20+ Merkmale)
- `y`: Zielvariable Series (binär: 0/1)

**Datenfluss:**
```
OpenML API / CSV-Datei
        ↓
CreditDataLoader
        ↓
Roh-DataFrame (1000 × 20+)
        ↓
Gespeichert in data/raw/
```

### 2. Vorverarbeitungsschicht (`preprocessing.py`)

**Verantwortlichkeiten:**
- Fehlende Werte behandeln (Median/Modus-Imputation)
- Kategoriale Variablen kodieren (One-Hot-Encoding)
- Features skalieren (StandardScaler)
- **PCA anwenden** (Dimensionsreduktion - KRITISCH für QML)
- Train/Test-Split mit Stratifizierung

**Wichtige Klassen:**
- `CreditDataPreprocessor`: Komplette Vorverarbeitungs-Pipeline

**Pipeline-Stufen:**
```
Rohdaten (1000 × 20+)
        ↓
Behandlung fehlender Werte
        ↓
Kategoriales Encoding (One-Hot)
        ↓
Train/Test-Split (80/20)
        ↓
Feature-Skalierung (StandardScaler)
        ↓
PCA-Reduktion (n → 4 Standard)
        ↓
Finale Daten (800 × 4 Train, 200 × 4 Test)
```

**Warum PCA kritisch ist:**
- Quantensimulatoren sind durch Qubits begrenzt
- 1 Feature = 1 Qubit in Quanten-Feature-Map
- Muss ~60 kodierte Features → 4-8 Features reduzieren
- Erhält 60-80% Varianz mit 4-8 Komponenten

### 3. Klassische SVM-Schicht (`classical_svm.py`)

**Verantwortlichkeiten:**
- Klassische SVM mit verschiedenen Kerneln trainieren
- Modellleistung evaluieren
- Konfusionsmatrix und ROC-Kurven generieren
- Verschiedene Kernel vergleichen
- Hyperparameter-Tuning

**Wichtige Klassen:**
- `ClassicalSVM`: Haupt-SVM-Wrapper mit Evaluation

**Unterstützte Kernel:**
- **Linear**: `K(x, y) = x^T y`
- **RBF** (Standard): `K(x, y) = exp(-γ||x-y||²)`
- **Polynomial**: `K(x, y) = (γx^T y + r)^d`
- **Sigmoid**: `K(x, y) = tanh(γx^T y + r)`

**Evaluationsmetriken:**
- Klassifikation: Genauigkeit, Präzision, Recall, F1-Wert
- Ranking: ROC AUC
- Rechenleistung: Trainingszeit, Vorhersagezeit
- Modell: Anzahl der Support-Vektoren

### 4. Quantum SVM-Schicht (`quantum_svm.py`)

**Verantwortlichkeiten:**
- Quanten-Feature-Map implementieren
- Quanten-Kernel-Schätzung
- Integration mit klassischer SVM
- Vergleich mit klassischer Leistung

**Architektur:**
```
Vorverarbeitete Daten (n × 4)
        ↓
Quanten-Feature-Map (ZZFeatureMap)
        ↓
Quanten-Kernel-Matrix-Schätzung
        ↓
Klassische SVM mit Quanten-Kernel
        ↓
Vorhersagen & Evaluation
```

**Quanten-Komponenten:**
- **Feature-Map**: Kodiert klassische Daten in Quantenzustände
- **Kernel-Schätzung**: Berechnet Kernel-Matrix mittels Quantenschaltungen
- **Simulator**: Qiskit's QASM-Simulator oder echte Quantenhardware

## Datenfluss-Architektur

```
┌─────────────────┐
│   OpenML API    │
│   Dataset #31   │
└────────┬────────┘
         │ load_from_openml()
         ▼
┌─────────────────┐
│  Roh-DataFrame  │
│  1000 × 20+     │
└────────┬────────┘
         │ preprocess_data()
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Kodierte Daten │──────│  Transformatoren:│
│  1000 × ~60     │      │  - ColumnTransf. │
└────────┬────────┘      │  - StandardScaler│
         │                └──────────────────┘
         │ train_test_split()
         ▼
┌──────────────────┐    ┌──────────────────┐
│  Train: 800 × 60 │    │  Test: 200 × 60  │
└────────┬─────────┘    └────────┬─────────┘
         │                        │
         │ apply_pca()            │ apply_pca()
         ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│  Train: 800 × 4  │    │  Test: 200 × 4   │
└────────┬─────────┘    └────────┬─────────┘
         │                        │
         │ train()                │
         ▼                        │
┌─────────────────┐              │
│  Trainiertes    │              │
│  SVM-Modell     │              │
└────────┬────────┘              │
         │ predict()              │
         │◄───────────────────────┘
         ▼
┌─────────────────┐
│  Vorhersagen    │
│  & Metriken     │
└─────────────────┘
```

## Dateiorganisation

```
qml-credit-risk-benchmark/
│
├── src/                          # Kernimplementierung
│   ├── __init__.py               # Paket-Initialisierung
│   ├── data_loader.py            # Datenladen (OpenML/CSV)
│   ├── preprocessing.py          # Vollständige Vorverarbeitungs-Pipeline
│   ├── classical_svm.py          # Klassische SVM-Implementierung
│   └── quantum_svm.py            # Quantum SVM-Implementierung
│
├── data/                         # Datenspeicherung
│   ├── raw/                      # Rohdaten
│   │   └── german_credit.csv    # Gecachter Datensatz
│   └── processed/                # Vorverarbeitete Daten
│
├── models/                       # Gespeicherte Modelle
│   ├── preprocessor.pkl          # Angepasster Präprozessor
│   ├── classical_svm.pkl         # Trainiertes klassisches Modell
│   └── quantum_svm.pkl           # Trainiertes Quantum-Modell
│
├── results/                      # Ausgaben
│   ├── confusion_matrix_*.png   # Konfusionsmatrizen
│   ├── roc_curve_*.png          # ROC-Kurven
│   └── comparison_results.csv   # Vergleichstabelle
│
├── notebooks/                    # Jupyter Notebooks
│   └── 01_classical_svm_*.ipynb # Interaktive Exploration
│
├── main.py                       # Hauptausführungsskript
├── config.py                     # Konfigurationseinstellungen
├── test_installation.py          # Installationsverifikation
├── requirements.txt              # Abhängigkeiten
├── README.md                     # Vollständige Dokumentation
├── QUICKSTART.md                 # Schnellstart-Anleitung
└── ARCHITECTURE.md               # Diese Datei
```

## Designprinzipien

### 1. Modularität
Jede Komponente ist unabhängig und kann einzeln verwendet werden:
```python
# Nur den Data Loader verwenden
from src.data_loader import load_credit_data
X, y = load_credit_data("openml")

# Nur den Präprozessor verwenden
from src.preprocessing import CreditDataPreprocessor
prep = CreditDataPreprocessor(n_components=4)
```

### 2. Konfigurierbarkeit
Alle Parameter sind exponiert und konfigurierbar:
- PCA-Komponenten (`n_components`)
- Testgröße (`test_size`)
- SVM-Hyperparameter (`C`, `gamma`, `kernel`)
- Random State für Reproduzierbarkeit

### 3. Reproduzierbarkeit
- Fixierte Random Seeds
- Gespeicherte Präprozessoren und Modelle
- Deterministische Train/Test-Splits

### 4. Erweiterbarkeit
Einfaches Hinzufügen neuer Komponenten:
- Neue Kernel
- Neue Vorverarbeitungsschritte
- Neue Evaluationsmetriken
- Quantum-Implementierung

### 5. Typsicherheit
Alle Funktionen verwenden Type Hints:
```python
def preprocess_data(
    self,
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...
```

## Leistungsüberlegungen

### Klassische SVM
- **Trainingszeit**: O(n² bis n³) wobei n = Anzahl der Stichproben
- **Speicher**: O(n²) für Kernel-Matrix
- **Inferenz**: O(n_sv × n_features) wobei n_sv = Support-Vektoren

**Optimierungen:**
- Linearer Kernel für große Datensätze
- PCA reduziert den Feature-Raum
- Stratifizierte Stichprobenziehung erhält Klassenbalance

### Quantum SVM
- **Trainingszeit**: Deutlich länger aufgrund der Quantensimulation
- **Quantenschaltungen**: O(n_qubits × Tiefe)
- **Shots**: Mehrere Durchläufe für Quantenmessungen benötigt

**Trade-offs:**
- Klassisch: Schnell, begrenzt auf polynomiale Kernel
- Quantum: Langsam in Simulation, Zugang zum exponentiellen Hilbert-Raum

## Teststrategie

### Unit Tests (Pro Modul)
```python
# Data Loader testen
python src/data_loader.py

# Präprozessor testen
python src/preprocessing.py

# Klassische SVM testen
python src/classical_svm.py
```

### Integrationstest
```python
# Vollständiger Pipeline-Test
python test_installation.py
```

### End-to-End-Test
```bash
# Kompletter Workflow
python main.py --mode classical
```

## Quantum-Implementierungs-Roadmap

### Phase 3A: Basis-QSVM
1. Quanten-Feature-Map implementieren (ZZFeatureMap)
2. Quanten-Kernel-Schätzung
3. Integration mit sklearn SVC

### Phase 3B: Optimierung
1. Schaltungsoptimierung
2. Parameter-Tuning
3. Hardware-Tests (falls verfügbar)

### Phase 4: Vergleich
1. Leistungsmetriken-Vergleich
2. Rechenkosten-Analyse
3. Skalierbarkeitsstudie

## Referenzen

**Klassische SVM:**
- Scikit-learn Dokumentation
- "Pattern Recognition and Machine Learning" von Bishop

**Quantum SVM:**
- Havlíček et al. "Supervised learning with quantum-enhanced feature spaces" (2019)
- Qiskit Machine Learning Dokumentation
- PennyLane Tutorials

---

**Version**: 0.1.0 (Klassische Implementierung abgeschlossen)
**Zuletzt aktualisiert**: Januar 2025
