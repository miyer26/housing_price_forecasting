import pytest
import pandas as pd
from src.data_ingestion import DataIngestion
import os


class TestDataIngestion:
    """Test suite for DataIngestion class"""

    @pytest.fixture
    def data_ingestion(self):
        """Fixture to create DataIngestion instance for tests"""
        return DataIngestion()

    def test_import_csv_valid_file(self, data_ingestion, tmp_path):
        """Test importing a valid CSV file"""
        # Create a test CSV file
        test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        test_file = tmp_path / "test.csv"
        test_df.to_csv(test_file, index=False)

        # Test the import
        result = data_ingestion.import_csv(str(test_file))
        assert isinstance(result, pd.DataFrame)
        assert result.equals(test_df)

    def test_import_csv_nonexistent_file(self, data_ingestion):
        """Test importing a non-existent file"""
        result = data_ingestion.import_csv("nonexistent.csv")
        assert result is None

    def test_import_csv_empty_file(self, data_ingestion, tmp_path):
        """Test importing an empty CSV file"""
        # Create empty file
        test_file = tmp_path / "empty.csv"
        test_file.write_text("")

        result = data_ingestion.import_csv(str(test_file))
        assert result is None

    def test_import_csv_invalid_path(self, data_ingestion):
        """Test importing with invalid path"""
        result = data_ingestion.import_csv("")
        assert result is None
