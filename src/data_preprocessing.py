import pandas as pd
import logging
from typing import Optional, Dict, List
from .preprocessing.imputation import DataImputation
from .preprocessing.cleaner import DataCleaner
from .preprocessing.outlier_detector import OutlierDetector
from .data_ingestion import DataIngestion
from .preprocessing.encoder import DataEncoder


class DataPreprocessor:
    """Main class for orchestrating all data preprocessing steps"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.data_ingestion = DataIngestion()
        self.data: Optional[pd.DataFrame] = None

    def load_data(self, file_path: str) -> bool:
        """Load data using DataIngestion class"""
        self.data = self.data_ingestion.import_csv(file_path)
        return self.data is not None

    def preprocess(self, config: Dict) -> Optional[pd.DataFrame]:
        """
        Execute all preprocessing steps based on configuration

        Args:
            config: Dictionary containing preprocessing configuration
            {
                'drop_columns': List[str],
                'na_threshold': float,
                'median_imputation': List[str],
                'mean_imputation': List[str],
                'mode_imputation': List[str],
                'outlier_columns': Dict[str, Dict],  # {column: {'method': str, 'threshold': float}}
                'categorical_encoding': Dict[str, str],  # {column: encoding_method}
                'feature_scaling': Dict[str, str]  # {column: scaling_method}
            }
        """
        if self.data is None:
            self.logger.error("No data loaded. Call load_data first.")
            return None

        try:
            processed_data = self.data.copy()

            # 1. Initial Cleaning Steps
            self.logger.info("Starting initial cleaning steps...")
            cleaner = DataCleaner(processed_data)

            if "drop_columns" in config:
                processed_data = cleaner.drop_columns(config["drop_columns"])
            if "na_threshold" in config:
                processed_data = cleaner.drop_na_columns(config["na_threshold"])

            # Handle imputation
            imputation_dict = {
                **{col: "median" for col in config.get("median_imputation", [])},
                **{col: "mean" for col in config.get("mean_imputation", [])},
                **{col: "mode" for col in config.get("mode_imputation", [])},
            }
            if imputation_dict:
                processed_data = cleaner.impute_values(imputation_dict)

            # 2. Outlier Detection and Handling
            self.logger.info("Processing outliers...")
            if "outlier_columns" in config:
                outlier_detector = OutlierDetector(processed_data)
                for column, settings in config["outlier_columns"].items():
                    processed_data = outlier_detector.remove_outliers(
                        column,
                        method=settings.get("method", "iqr"),
                        threshold=settings.get("threshold", 1.5),
                    )  # type: ignore

            # 3. Categorical Encoding
            if "categorical_encoding" in config:
                encoder = DataEncoder(processed_data)
                for column, method in config["categorical_encoding"].items():
                    processed_data = encoder.encode_categorical(column, method)

            # 4. Feature Scaling
            if "feature_scaling" in config:
                encoder = DataEncoder(
                    processed_data
                )  # Create new encoder with latest data
                for column, method in config["feature_scaling"].items():
                    processed_data = encoder.scale_features(column, method)

            self.logger.info("Preprocessing completed successfully")
            self.data = processed_data
            return self.data

        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}")
            return None
