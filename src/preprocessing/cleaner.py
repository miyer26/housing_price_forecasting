import pandas as pd
from typing import List, Dict
from utils.logger import get_logger


class DataCleaner:
    """
    Handles initial data cleaning steps: missing values and NA handling.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Args:
            data (pd.DataFrame): Input dataframe to clean
        """
        self.data = data.copy()
        self.logger = get_logger(__name__)
        self.logger.info("Initializing DataCleaner")
        self.logger.info(f"Initial data shape: {self.data.shape}")

    def drop_na_columns(self, threshold: float = 50.0) -> pd.DataFrame:
        """
        Drop columns with NA values above threshold percentage.

        Args:
            threshold (float): Percentage threshold for dropping columns
        """
        self.logger.info(f"Dropping columns with >{threshold}% missing values")
        missing_percentages = self.data.isna().mean() * 100
        cols_to_drop = missing_percentages[
            missing_percentages > threshold
        ].index.tolist()

        if cols_to_drop:
            self.data.drop(columns=cols_to_drop, inplace=True)
            self.logger.info(f"Dropped columns: {cols_to_drop}")
        else:
            self.logger.info("No columns met the threshold for dropping")

        return self.data

    def drop_columns(self, columns: List[str]) -> pd.DataFrame:
        """
        Drop specified columns from the dataframe.

        Args:
            columns (List[str]): List of columns to drop
        """
        existing_cols = [col for col in columns if col in self.data.columns]
        if existing_cols:
            self.data.drop(columns=existing_cols, inplace=True)
            print(f"Dropped columns: {existing_cols}")

        return self.data

    def impute_values(self, columns_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Impute missing values in specified columns.

        Args:
            columns_dict (Dict[str, str]): Dictionary of {column: strategy}
                where strategy can be 'mean', 'median', or 'mode'
        """
        for column, strategy in columns_dict.items():
            if column not in self.data.columns:
                print(f"Warning: Column '{column}' not found in dataframe")
                continue

            if strategy == "mean":
                value = self.data[column].mean()
            elif strategy == "median":
                value = self.data[column].median()
            elif strategy == "mode":
                value = self.data[column].mode()[0]
            else:
                raise ValueError(f"Unknown imputation strategy: {strategy}")

            self.data[column].fillna(value, inplace=True)
            print(f"Imputed {column} using {strategy} strategy")

        return self.data

    def get_cleaned_data(self):
        """
        Returns the cleaned dataframe.

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        return self.data
