import pandas as pd
import logging
from typing import Optional, Dict, List
from .preprocessing.imputation import DataImputation
from .preprocessing.cleaner import DataCleaner
from .data_ingestion import DataIngestion


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
                'mode_imputation': List[str]
            }
        """
        if self.data is None:
            self.logger.error("No data loaded. Call load_data first.")
            return None

        try:
            # Initialize preprocessing classes
            cleaner = DataCleaner(self.data)
            imputer = DataImputation(self.data)

            # Cleaning steps
            cleaner.drop_duplicates()
            if "drop_columns" in config:
                cleaner.remove_columns(config["drop_columns"])
            if "na_threshold" in config:
                cleaner.drop_nas(config["na_threshold"])

            # Imputation steps
            for column in config.get("median_imputation", []):
                imputer.fill_median(column)
            for column in config.get("mean_imputation", []):
                imputer.fill_mean(column)
            for column in config.get("mode_imputation", []):
                imputer.fill_mode(column)

            return self.data

        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}")
            return None
