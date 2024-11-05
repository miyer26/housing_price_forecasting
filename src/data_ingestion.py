import pandas as pd
import logging
import os
from typing import Optional


class DataIngestion:
    """
    A class for handling data ingestion operations.

    This class provides methods for importing data from various sources
    with proper error handling and logging.

    Attributes:
        logger (logging.Logger): Logger instance for tracking operations
    """

    def __init__(self) -> None:
        """Initialize the DataIngestion class with a logger."""
        self.logger = logging.getLogger(__name__)

    def import_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Import data from a CSV file into a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file to import

        Returns:
            Optional[pd.DataFrame]: DataFrame containing the imported data if successful,
                                  None otherwise

        Raises:
            FileNotFoundError: If the specified file does not exist
            pd.errors.EmptyDataError: If the file is empty
            pd.errors.ParserError: If the file cannot be parsed as CSV
            AssertionError: If file_path is not a string or is empty
        """
        # Input validation
        try:
            assert isinstance(file_path, str), "file_path must be a string"
            assert len(file_path.strip()) > 0, "file_path cannot be empty"
        except AssertionError as e:
            self.logger.error(f"Validation error: {e}")
            return None

        try:
            # Verify file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Attempt to read the CSV file
            df = pd.read_csv(file_path)

            # Check if DataFrame is empty
            if df.empty:
                self.logger.warning("Imported DataFrame is empty")

            # Verify DataFrame has at least one column
            assert len(df.columns) > 0, "CSV file must contain at least one column"

            self.logger.info(f"Successfully imported data from {file_path}")
            return df

        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {e}")
            return None
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"Empty file error: {e}")
            return None
        except pd.errors.ParserError as e:
            self.logger.error(f"CSV parsing error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error importing data: {e}")
            return None
