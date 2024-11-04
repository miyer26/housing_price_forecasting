import pandas as pd
import logging
import os


def import_data(file_path: str) -> pd.DataFrame:
    """
    Import data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file to import

    Returns:
        pd.DataFrame: DataFrame containing the imported data if successful, None otherwise

    Raises:
        FileNotFoundError: If the specified file does not exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed as CSV
        AssertionError: If file_path is not a string or is empty
    """
    logger = logging.getLogger(__name__)

    # Input validation
    assert isinstance(file_path, str), "file_path must be a string"
    assert len(file_path.strip()) > 0, "file_path cannot be empty"

    try:
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Attempt to read the CSV file
        df = pd.read_csv(file_path)

        # Check if DataFrame is empty
        if df.empty:
            logger.warning("Imported DataFrame is empty")

        # Verify DataFrame has at least one column
        assert len(df.columns) > 0, "CSV file must contain at least one column"

        logger.info(f"Successfully imported data from {file_path}")
        return df

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        return None
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty file error: {e}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        return None
    except AssertionError as e:
        logger.error(f"Validation error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error importing data: {e}")
        return None
