import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from utils.logger import get_logger


class OutlierDetector:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.logger = get_logger(__name__)
        self.logger.info("Initializing OutlierDetector")
        self.logger.info(f"Initial data shape: {self.data.shape}")

    def detect_iqr_outliers(
        self, column: str, threshold: float = 1.5
    ) -> Tuple[pd.Series, pd.Series]:
        self.logger.info(
            f"Detecting IQR outliers in column {column} with threshold {threshold}"
        )
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            msg = f"Column '{column}' must be numeric"
            self.logger.error(msg)
            raise ValueError(msg)

        # ... rest of the method with logging

    def detect_zscore_outliers(self, column: str, threshold: float = 3) -> pd.Series:
        """
        Detect outliers using the Z-score method.

        Args:
            column (str): Column to check for outliers
            threshold (float): Number of standard deviations for outlier detection
        """
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric")

        z_scores = np.abs(
            (self.data[column] - self.data[column].mean()) / self.data[column].std()
        )
        return self.data[z_scores > threshold][column]

    def detect_isolation_forest_outliers(
        self, columns: List[str], contamination: float = 0.1
    ) -> pd.Series:
        """
        Detect outliers using Isolation Forest.

        Args:
            columns (List[str]): Columns to use for outlier detection
            contamination (float): Expected proportion of outliers
        """
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(self.data[columns])
        return self.data[predictions == -1]

    def detect_robust_covariance_outliers(
        self, columns: List[str], contamination: float = 0.1
    ) -> pd.Series:
        """
        Detect outliers using Robust Covariance Estimation.

        Args:
            columns (List[str]): Columns to use for outlier detection
            contamination (float): Expected proportion of outliers
        """
        robust_cov = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = robust_cov.fit_predict(self.data[columns])
        return self.data[predictions == -1]

    def remove_outliers(
        self, column: str, method: str = "iqr", threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from the dataset.

        Args:
            column (str): Column to remove outliers from
            method (str): Method to use ('iqr', 'zscore', 'iforest', 'robust_cov')
            threshold (float): Threshold for outlier detection
        """
        if method == "iqr":
            lower_outliers, upper_outliers = self.detect_iqr_outliers(column, threshold)
            self.data = self.data[
                ~self.data[column].isin(pd.concat([lower_outliers, upper_outliers]))
            ]
        elif method == "zscore":
            outliers = self.detect_zscore_outliers(column, threshold)
            self.data = self.data[~self.data[column].isin(outliers)]
        elif method == "iforest":
            outliers = self.detect_isolation_forest_outliers([column], threshold)
            self.data = self.data[~self.data.index.isin(outliers.index)]
        elif method == "robust_cov":
            outliers = self.detect_robust_covariance_outliers([column], threshold)
            self.data = self.data[~self.data.index.isin(outliers.index)]

        return self.data
