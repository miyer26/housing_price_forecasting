import pandas as pd
from typing import Optional


class DataImputation:
    """A class for handling different types of data imputation"""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def fill_median(self, column: str) -> None:
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric for median imputation")
        self.data[column].fillna(self.data[column].median(), inplace=True)

    def fill_mean(self, column: str) -> None:
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' must be numeric for mean imputation")
        self.data[column].fillna(self.data[column].mean(), inplace=True)

    def fill_mode(self, column: str) -> None:
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")
        mode_value = self.data[column].mode()[0]
        self.data[column].fillna(mode_value, inplace=True)
