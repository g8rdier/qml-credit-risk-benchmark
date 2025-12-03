"""
Data Loading Module
Handles loading of the German Credit Risk dataset from various sources.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml


class CreditDataLoader:
    """
    Loads and provides access to the German Credit Risk dataset.

    Can load from:
    - OpenML (default, dataset_id=31)
    - Local CSV file
    - Kaggle API (if credentials configured)
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory where raw data is stored
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series] = None

    def load_from_openml(self, dataset_name: str = "credit-g") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load German Credit Data from OpenML.

        Args:
            dataset_name: OpenML dataset name ("credit-g" = German Credit Data)

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        print(f"📥 Loading dataset from OpenML (name: {dataset_name})...")

        # Fetch the dataset
        data = fetch_openml(name=dataset_name, version=1, as_frame=True, parser='auto')

        self.df = data.data
        self.target = data.target

        # Convert target to binary (good=1, bad=0)
        self.target = (self.target == 'good').astype(int)

        print(f"✅ Loaded {len(self.df)} samples with {len(self.df.columns)} features")
        print(f"   Target distribution: {self.target.value_counts().to_dict()}")

        return self.df.copy(), self.target.copy()

    def load_from_kaggle(self, dataset_id: str = "kabure/german-credit-data-with-risk") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load German Credit Data from Kaggle using kagglehub.

        Args:
            dataset_id: Kaggle dataset identifier (default: "kabure/german-credit-data-with-risk")

        Returns:
            Tuple of (features DataFrame, target Series)

        Note:
            This is more reliable than OpenML as Kaggle has better infrastructure.
            The dataset is cached locally after first download.
        """
        print(f"📥 Loading dataset from Kaggle (ID: {dataset_id})...")

        try:
            import kagglehub

            # Download dataset (cached locally after first download)
            path = kagglehub.dataset_download(dataset_id)
            print(f"   Dataset cached at: {path}")

            # Find the CSV file in the downloaded directory
            from pathlib import Path
            dataset_path = Path(path)

            # Look for german_credit_data.csv or similar
            csv_files = list(dataset_path.glob("*.csv"))

            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {path}")

            # Use the first CSV file found
            csv_file = csv_files[0]
            print(f"   Loading from: {csv_file.name}")

            # Read the CSV
            df = pd.read_csv(csv_file)

            # Kaggle version has specific column structure
            # Typically: features + 'Risk' column (or similar)
            if 'Risk' in df.columns:
                self.target = (df['Risk'] == 'good').astype(int)
                self.df = df.drop('Risk', axis=1)
            elif 'class' in df.columns:
                self.target = (df['class'] == 'good').astype(int)
                self.df = df.drop('class', axis=1)
            else:
                # Assume last column is target
                self.target = df.iloc[:, -1]
                self.df = df.iloc[:, :-1]

                # Convert if necessary
                if self.target.dtype == 'object':
                    unique_values = self.target.unique()
                    if len(unique_values) == 2:
                        self.target = (self.target == unique_values[0]).astype(int)

            print(f"✅ Loaded {len(self.df)} samples with {len(self.df.columns)} features")
            print(f"   Target distribution: {self.target.value_counts().to_dict()}")

            return self.df.copy(), self.target.copy()

        except ImportError:
            raise ImportError(
                "kagglehub is not installed. Install it with: pixi add kagglehub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load from Kaggle: {e}")

    def load_from_csv(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load credit data from a local CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            Tuple of (features DataFrame, target Series)

        Note:
            Assumes the last column is the target variable.
            Adjust if your CSV has a different structure.
        """
        print(f"📥 Loading dataset from CSV: {filepath}...")

        df = pd.read_csv(filepath)

        # Assuming last column is target
        self.target = df.iloc[:, -1]
        self.df = df.iloc[:, :-1]

        # If target is not numeric, convert it
        if self.target.dtype == 'object':
            # Map good/bad or similar labels to 1/0
            unique_values = self.target.unique()
            if len(unique_values) == 2:
                self.target = (self.target == unique_values[0]).astype(int)

        print(f"✅ Loaded {len(self.df)} samples with {len(self.df.columns)} features")
        print(f"   Target distribution: {self.target.value_counts().to_dict()}")

        return self.df.copy(), self.target.copy()

    def get_data_summary(self) -> dict:
        """
        Get a summary of the loaded dataset.

        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_from_openml() or load_from_csv() first.")

        return {
            "n_samples": len(self.df),
            "n_features": len(self.df.columns),
            "n_numeric": self.df.select_dtypes(include=[np.number]).shape[1],
            "n_categorical": self.df.select_dtypes(include=['object', 'category']).shape[1],
            "missing_values": self.df.isnull().sum().sum(),
            "target_distribution": self.target.value_counts().to_dict(),
            "feature_names": list(self.df.columns)
        }

    def save_raw_data(self, filename: str = "german_credit.csv") -> Path:
        """
        Save the loaded data to CSV for reproducibility.

        Args:
            filename: Name of the output CSV file

        Returns:
            Path to saved file
        """
        if self.df is None:
            raise ValueError("No data loaded.")

        output_path = self.data_dir / filename

        # Combine features and target
        full_df = self.df.copy()
        full_df['target'] = self.target

        full_df.to_csv(output_path, index=False)
        print(f"💾 Saved raw data to: {output_path}")

        return output_path


def load_credit_data(source: str = "kaggle") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load credit data with automatic fallback.

    Args:
        source: Data source - "kaggle" (default, most reliable), "openml", or path to CSV file

    Returns:
        Tuple of (features DataFrame, target Series)

    Examples:
        >>> X, y = load_credit_data("kaggle")      # Use Kaggle (recommended)
        >>> X, y = load_credit_data("openml")      # Use OpenML (may be unreliable)
        >>> X, y = load_credit_data("data.csv")    # Use local CSV file
    """
    loader = CreditDataLoader()

    if source.lower() == "kaggle":
        return loader.load_from_kaggle()
    elif source.lower() == "openml":
        return loader.load_from_openml()
    else:
        # Assume it's a file path
        return loader.load_from_csv(source)


if __name__ == "__main__":
    # Test the data loader
    loader = CreditDataLoader()
    print("Testing Kaggle data loader (recommended source)...\n")
    X, y = loader.load_from_kaggle()

    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)

    summary = loader.get_data_summary()
    for key, value in summary.items():
        if key != "feature_names":
            print(f"{key}: {value}")

    print("\nFirst few samples:")
    print(X.head())

    # Save for future use
    loader.save_raw_data()
