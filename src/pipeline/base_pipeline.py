from abc import ABC, abstractmethod
from typing import Optional, Dict
import pandas as pd
from ..utils.logger import get_logger
from ..data_ingestion import DataIngestion


class BasePipeline(ABC):
    """Abstract base class for all pipelines"""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.data: Optional[pd.DataFrame] = None

    @abstractmethod
    def run(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """Main pipeline execution method"""
        pass

    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Common data loading functionality"""
        try:
            ingestion = DataIngestion()
            self.data = ingestion.import_csv(file_path)
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return None
