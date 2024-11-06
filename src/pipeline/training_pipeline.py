import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from sklearn.model_selection import train_test_split

# Local imports
from .base_pipeline import BasePipeline
from ..data_preprocessing import DataPreprocessor
from ..utils.logger import get_logger


class TrainingPipeline(BasePipeline):
    """Pipeline for data preprocessing and model training"""

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor = DataPreprocessor()

    def run(
        self,
        file_path: str,
        preprocess_config: Dict,
        feature_config: Dict,
        encoder_config: Dict,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Run the complete training pipeline

        Args:
            file_path: Path to input data
            preprocess_config: Configuration for preprocessing steps
            feature_config: Configuration for feature engineering
            encoder_config: Configuration for encoding
            target_column: Name of target variable (if any)
            test_size: Size of test split

        Returns:
            If target_column is provided: Tuple of (X_train, X_test, y_train, y_test)
            Otherwise: Processed DataFrame
        """
        try:
            # 1. Load Data
            if not self.preprocessor.load_data(file_path):
                return None

            # 2. Run Preprocessing
            self.logger.info("Starting preprocessing...")
            processed_data = self.preprocessor.preprocess(preprocess_config)
            if processed_data is None:
                return None

            self.data = processed_data
            self.logger.info("Preprocessing completed successfully")

            # 3. Train/Test Split if target is provided
            if target_column and target_column in self.data.columns:
                X = self.data.drop(columns=[target_column])
                y = self.data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                return X_train, X_test, y_train, y_test

            return self.data

        except Exception as e:
            self.logger.error(f"Error in training pipeline: {str(e)}")
            return None
