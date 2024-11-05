from typing import Dict, Union, Optional, List
import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.logger import get_logger


class EDA:
    """
    A class for performing exploratory data analysis on pandas DataFrames.

    This class provides methods for calculating statistics and generating
    various visualization plots to help understand the underlying data structure.

    Attributes:
        data (pd.DataFrame): The input DataFrame to analyze
        logger (logging.Logger): Logger instance for tracking operations
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the EDA class with a pandas DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to analyze
        """
        self.data = data.copy()
        self.logger = get_logger(__name__)
        self.logger.info("Initializing EDA")
        self.logger.info(f"Dataset shape: {self.data.shape}")
        self.logger.info(f"Columns: {list(self.data.columns)}")

    def calculate_statistics(
        self, column: pd.Series
    ) -> Dict[str, Union[float, int, bool, None]]:
        """
        Calculate comprehensive statistics for a given column.

        Args:
            column (pd.Series): The column to analyze

        Returns:
            Dict[str, Union[float, int, bool, None]]: Dictionary containing various statistics
                including mean, median, standard deviation, etc.
        """
        stats_dict = {
            "Mean": column.mean(),
            "Median": column.median(),
            "Std Dev": column.std(),
            "Min": column.min(),
            "Max": column.max(),
            "Count": column.count(),
            "Unique Values": column.nunique(),
            "Is Numeric": pd.api.types.is_numeric_dtype(column),
            "Missing Percentage": round(column.isna().mean() * 100, 2),
            "Skewness": stats.skew(column)
            if pd.api.types.is_numeric_dtype(column)
            else None,
            "Kurtosis": stats.kurtosis(column)
            if pd.api.types.is_numeric_dtype(column)
            else None,
            "IQR": column.quantile(0.75) - column.quantile(0.25)
            if pd.api.types.is_numeric_dtype(column)
            else None,
        }
        return stats_dict

    def plot_distribution(self, output_dir: str = "EDA"):
        """
        Plot and save distribution plots for numeric columns.

        Args:
            output_dir (str): Directory to save the plots (default: 'EDA')
        """
        # Create output directory if it doesn't exist
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            return

        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                try:
                    plt.figure(figsize=(10, 6))
                    plt.title(f"Distribution of {col}")
                    plt.hist(
                        self.data[col], bins=50, color="skyblue", edgecolor="black"
                    )
                    plt.xlabel("Value")
                    plt.ylabel("Frequency")

                    # Save plot
                    plot_path = os.path.join(output_dir, f"{col}_distribution.png")
                    plt.savefig(plot_path)
                    plt.close()

                    self.logger.info(f"Saved distribution plot for {col}")

                    # Calculate and log statistics
                    stats = self.calculate_statistics(self.data[col])
                    self.logger.info(f"Statistics for {col}: {stats}")

                except Exception as e:
                    self.logger.error(f"Error plotting distribution for {col}: {e}")

    def plot_correlation_matrix(self, output_dir: str):
        """Plot correlation matrix for numeric columns"""
        numeric_cols = [
            col
            for col in self.data.columns
            if pd.api.types.is_numeric_dtype(self.data[col])
        ]
        numeric_data = self.data[numeric_cols]
        if not numeric_data.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", center=0)
            plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
            plt.close()

    def plot_missing_values(self, output_dir: str):
        """Plot missing values heatmap"""
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), yticklabels=False, cbar=True, cmap="viridis")
        plt.savefig(os.path.join(output_dir, "missing_values.png"))
        plt.close()

    def plot_boxplots(self, output_dir: str):
        """Plot boxplots for numeric columns"""
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.data[col])
                plt.title(f"Boxplot of {col}")
                plt.savefig(os.path.join(output_dir, f"{col}_boxplot.png"))
                plt.close()

    def analyze_missing_values(self) -> Dict[str, float]:
        """
        Analyze missing values in the dataset.

        Returns:
            Dict[str, float]: Dictionary of column names and their missing value percentages
        """
        self.logger.info("Analyzing missing values")
        missing_percentages = (self.data.isna().mean() * 100).round(2)

        # Log columns with missing values
        columns_with_missing = missing_percentages[missing_percentages > 0]
        if not columns_with_missing.empty:
            self.logger.info("Columns with missing values:")
            for col, pct in columns_with_missing.items():
                self.logger.info(f"- {col}: {pct}% missing")
        else:
            self.logger.info("No missing values found in the dataset")

        return missing_percentages.to_dict()

    def analyze_data_types(self) -> Dict[str, str]:
        """
        Analyze data types of columns.

        Returns:
            Dict[str, str]: Dictionary of column names and their data types
        """
        self.logger.info("Analyzing data types")
        dtypes = self.data.dtypes.astype(str).to_dict()

        for col, dtype in dtypes.items():
            self.logger.info(f"Column '{col}': {dtype}")

        return dtypes

    def analyze_numeric_columns(
        self, columns: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Analyze numeric columns statistics.

        Args:
            columns: Optional list of numeric columns to analyze

        Returns:
            Dict containing statistics for each numeric column
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns

        self.logger.info(f"Analyzing numeric columns: {list(columns)}")

        stats = {}
        for col in columns:
            if col not in self.data.columns:
                self.logger.warning(f"Column '{col}' not found in dataset")
                continue

            if not pd.api.types.is_numeric_dtype(self.data[col]):
                self.logger.warning(f"Column '{col}' is not numeric")
                continue

            stats[col] = {
                "mean": self.data[col].mean(),
                "median": self.data[col].median(),
                "std": self.data[col].std(),
                "skew": self.data[col].skew(),
                "kurtosis": self.data[col].kurtosis(),
            }

            self.logger.info(f"Statistics for {col}:")
            for metric, value in stats[col].items():
                self.logger.info(f"- {metric}: {value:.2f}")

            # Log potential outlier information
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = self.data[
                (self.data[col] < q1 - 1.5 * iqr) | (self.data[col] > q3 + 1.5 * iqr)
            ][col]

            if not outliers.empty:
                self.logger.info(f"Potential outliers in {col}: {len(outliers)} points")

    def analyze_categorical_columns(
        self, columns: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Analyze categorical columns.

        Args:
            columns: Optional list of categorical columns to analyze

        Returns:
            Dict containing statistics for each categorical column
        """
        if columns is None:
            columns = self.data.select_dtypes(include=["object", "category"]).columns

        self.logger.info(f"Analyzing categorical columns: {list(columns)}")

        stats = {}
        for col in columns:
            if col not in self.data.columns:
                self.logger.warning(f"Column '{col}' not found in dataset")
                continue

            value_counts = self.data[col].value_counts()
            unique_count = len(value_counts)

            stats[col] = {
                "unique_values": unique_count,
                "top_categories": value_counts.head().to_dict(),
            }

            self.logger.info(f"Statistics for {col}:")
            self.logger.info(f"- Unique values: {unique_count}")
            self.logger.info("- Top categories:")
            for cat, count in value_counts.head().items():
                self.logger.info(f"  * {cat}: {count}")

        return stats

    def plot_distributions(
        self, columns: Optional[List[str]] = None, save_path: Optional[str] = None
    ) -> None:
        """
        Plot distributions for specified columns.

        Args:
            columns: Optional list of columns to plot
            save_path: Optional path to save the plots
        """
        self.logger.info("Plotting distributions")

        if columns is None:
            columns = self.data.columns

        for col in columns:
            if col not in self.data.columns:
                self.logger.warning(f"Column '{col}' not found in dataset")
                continue

            plt.figure(figsize=(10, 6))

            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.logger.info(f"Creating histogram for numeric column: {col}")
                sns.histplot(data=self.data, x=col, kde=True)
            else:
                self.logger.info(f"Creating bar plot for categorical column: {col}")
                sns.countplot(data=self.data, x=col)

            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)

            if save_path:
                plt.savefig(f"{save_path}/distribution_{col}.png")
                self.logger.info(f"Saved plot for {col} to {save_path}")

            plt.close()
