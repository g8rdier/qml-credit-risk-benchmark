"""
Preprocessing Module
Handles data cleaning, encoding, scaling, and dimensionality reduction (PCA).

Critical for reducing features to a quantum-manageable number (4-8 qubits).
"""

from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import pickle
from pathlib import Path


class CreditDataPreprocessor:
    """
    Comprehensive preprocessor for credit risk data.

    Pipeline:
    1. Handle missing values
    2. Encode categorical variables
    3. Scale features (StandardScaler)
    4. Apply PCA for dimensionality reduction (critical for QML)
    """

    def __init__(self, n_components: int = 4, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the preprocessor.

        Args:
            n_components: Number of PCA components (= number of qubits for QSVM)
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state

        # Transformers (fitted during preprocessing)
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.column_transformer: Optional[ColumnTransformer] = None

        # Feature information
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_names_after_encoding: List[str] = []

        print(f"🔧 Initialized preprocessor:")
        print(f"   - Target PCA components: {n_components}")
        print(f"   - Test set size: {test_size * 100}%")
        print(f"   - Random state: {random_state}")

    def identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical features.

        Args:
            X: Input DataFrame

        Returns:
            Tuple of (numeric_features, categorical_features)
        """
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"\n📊 Feature Analysis:")
        print(f"   - Numeric features: {len(self.numeric_features)}")
        print(f"   - Categorical features: {len(self.categorical_features)}")

        return self.numeric_features, self.categorical_features

    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Strategy:
        - Numeric: Fill with median
        - Categorical: Fill with mode

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with missing values handled
        """
        X_clean = X.copy()
        missing_count = X_clean.isnull().sum().sum()

        if missing_count > 0:
            print(f"\n🧹 Handling {missing_count} missing values...")

            # Numeric features: fill with median
            for col in self.numeric_features:
                if X_clean[col].isnull().any():
                    median_val = X_clean[col].median()
                    X_clean[col].fillna(median_val, inplace=True)
                    print(f"   - {col}: filled {X[col].isnull().sum()} values with median ({median_val:.2f})")

            # Categorical features: fill with mode
            for col in self.categorical_features:
                if X_clean[col].isnull().any():
                    mode_val = X_clean[col].mode()[0]
                    X_clean[col].fillna(mode_val, inplace=True)
                    print(f"   - {col}: filled {X[col].isnull().sum()} values with mode ({mode_val})")
        else:
            print("\n✅ No missing values found")

        return X_clean

    def encode_and_scale(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Encode categorical variables and scale all features.

        Args:
            X: Input DataFrame
            fit: If True, fit the transformers. If False, use existing transformers.

        Returns:
            Scaled numpy array
        """
        if fit:
            print("\n🔄 Encoding categorical variables...")

            # Create column transformer
            transformers = []

            # Numeric features: just pass through (will scale later)
            if self.numeric_features:
                transformers.append(('num', 'passthrough', self.numeric_features))

            # Categorical features: one-hot encode
            if self.categorical_features:
                transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
                                     self.categorical_features))

            self.column_transformer = ColumnTransformer(
                transformers=transformers,
                remainder='drop'
            )

            # Fit and transform
            X_encoded = self.column_transformer.fit_transform(X)

            # Get feature names after encoding
            self.feature_names_after_encoding = self._get_feature_names_after_encoding()

            print(f"   - Features after encoding: {X_encoded.shape[1]}")

            # Scale the encoded features
            print("\n⚖️  Scaling features (StandardScaler)...")
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_encoded)

        else:
            # Transform only (use fitted transformers)
            if self.column_transformer is None or self.scaler is None:
                raise ValueError("Transformers not fitted. Call with fit=True first.")

            X_encoded = self.column_transformer.transform(X)
            X_scaled = self.scaler.transform(X_encoded)

        return X_scaled

    def apply_pca(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.

        This is CRITICAL for quantum implementation as it reduces
        the feature space to match the number of available qubits.

        Args:
            X: Input array (should be scaled)
            fit: If True, fit PCA. If False, use existing PCA.

        Returns:
            Transformed array with n_components features
        """
        if fit:
            print(f"\n🔬 Applying PCA (target: {self.n_components} components)...")

            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
            X_pca = self.pca.fit_transform(X)

            # Print explained variance
            explained_var = self.pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)

            print(f"   - Reduced from {X.shape[1]} to {X_pca.shape[1]} features")
            print(f"   - Explained variance per component:")
            for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
                print(f"     PC{i+1}: {var:.4f} (cumulative: {cum_var:.4f})")

        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call with fit=True first.")

            X_pca = self.pca.transform(X)

        return X_pca

    def _get_feature_names_after_encoding(self) -> List[str]:
        """Get feature names after one-hot encoding."""
        feature_names = []

        # Numeric features keep their names
        feature_names.extend(self.numeric_features)

        # Categorical features: get encoded names
        if self.categorical_features:
            cat_encoder = self.column_transformer.named_transformers_['cat']
            for i, cat_feature in enumerate(self.categorical_features):
                # Get categories (excluding first due to drop='first')
                categories = cat_encoder.categories_[i][1:]  # Skip first category
                feature_names.extend([f"{cat_feature}_{cat}" for cat in categories])

        return feature_names

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline.

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) - all after full preprocessing
        """
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)

        # 1. Identify feature types
        self.identify_feature_types(X)

        # 2. Handle missing values
        X_clean = self.handle_missing_values(X)

        # 3. Split data BEFORE encoding (to prevent data leakage)
        print(f"\n✂️  Splitting data ({(1-self.test_size)*100:.0f}% train / {self.test_size*100:.0f}% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        print(f"   - Train set: {len(X_train)} samples")
        print(f"   - Test set: {len(X_test)} samples")

        # 4. Encode and scale (fit on train, transform both)
        X_train_scaled = self.encode_and_scale(X_train, fit=True)
        X_test_scaled = self.encode_and_scale(X_test, fit=False)

        # 5. Apply PCA (fit on train, transform both)
        X_train_pca = self.apply_pca(X_train_scaled, fit=True)
        X_test_pca = self.apply_pca(X_test_scaled, fit=False)

        print("\n✅ Preprocessing complete!")
        print(f"   - Final feature space: {X_train_pca.shape[1]} dimensions")
        print(f"   - Ready for {X_train_pca.shape[1]}-qubit quantum implementation")

        return X_train_pca, X_test_pca, y_train.values, y_test.values

    def save_preprocessor(self, filepath: str = "models/preprocessor.pkl") -> Path:
        """
        Save fitted preprocessor for later use.

        Args:
            filepath: Output path for pickle file

        Returns:
            Path to saved file
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        preprocessor_state = {
            'scaler': self.scaler,
            'pca': self.pca,
            'column_transformer': self.column_transformer,
            'n_components': self.n_components,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_names_after_encoding': self.feature_names_after_encoding
        }

        with open(output_path, 'wb') as f:
            pickle.dump(preprocessor_state, f)

        print(f"\n💾 Saved preprocessor to: {output_path}")
        return output_path

    @classmethod
    def load_preprocessor(cls, filepath: str = "models/preprocessor.pkl") -> 'CreditDataPreprocessor':
        """
        Load a saved preprocessor.

        Args:
            filepath: Path to pickle file

        Returns:
            Loaded preprocessor instance
        """
        with open(filepath, 'rb') as f:
            preprocessor_state = pickle.load(f)

        # Create instance
        preprocessor = cls(n_components=preprocessor_state['n_components'])

        # Restore state
        preprocessor.scaler = preprocessor_state['scaler']
        preprocessor.pca = preprocessor_state['pca']
        preprocessor.column_transformer = preprocessor_state['column_transformer']
        preprocessor.numeric_features = preprocessor_state['numeric_features']
        preprocessor.categorical_features = preprocessor_state['categorical_features']
        preprocessor.feature_names_after_encoding = preprocessor_state['feature_names_after_encoding']

        print(f"📂 Loaded preprocessor from: {filepath}")
        return preprocessor


if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import load_credit_data

    # Load data
    X, y = load_credit_data("openml")

    # Preprocess with 4 components (for 4-qubit QSVM)
    preprocessor = CreditDataPreprocessor(n_components=4)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(X, y)

    print("\n" + "="*60)
    print("PREPROCESSED DATA SUMMARY")
    print("="*60)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Save preprocessor
    preprocessor.save_preprocessor()
