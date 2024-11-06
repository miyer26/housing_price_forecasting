from abc import ABC, abstractmethod
from typing import Optional, Dict
import pandas as pd
from ..utils.logger import get_logger
from ..data_ingestion import DataIngestion
from ..data_preprocessing import DataPreprocessor
from ..eda import EDA
from .base_pipeline import BasePipeline


class EDAPipeline(BasePipeline):
    """Pipeline for Exploratory Data Analysis"""

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor = DataPreprocessor()
        self.eda = None

    def run(
        self,
        file_path: str,
        basic_config: Dict,
        output_dir: str = "reports/eda",
        save_cleaned_data: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Run EDA pipeline

        Args:
            file_path: Path to raw data
            basic_config: Basic cleaning configuration
            output_dir: Directory to save EDA reports
            save_cleaned_data: Whether to save cleaned data for later use
        """
        try:
            # 1. Load Data
            if not self.load_data(file_path):
                return None

            # 2. Basic Cleaning
            self.logger.info("Performing basic cleaning...")
            clean_data = self.preprocessor.preprocess(basic_config)

            # 3. Run EDA
            self.logger.info("Running EDA analysis...")
            self.eda = EDA(clean_data)

            # Generate reports
            self.eda.plot_missing_values(output_dir)
            self.eda.plot_correlation_matrix(output_dir)
            self.eda.plot_boxplots(output_dir)
            self.eda.analyze_data_types()
            self.eda.plot_distribution(output_dir)
            self.eda.analyze_missing_values()
            self.eda.analyze_numeric_columns()
            self.eda.analyze_categorical_columns()

            # Save cleaned data if requested
            if save_cleaned_data:
                clean_data.to_csv(f"{output_dir}/cleaned_data.csv", index=False)

            return clean_data

        except Exception as e:
            self.logger.error(f"Error in EDA pipeline: {str(e)}")
            return None
