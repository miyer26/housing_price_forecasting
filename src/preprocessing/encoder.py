import logging
from typing import Optional
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)


class DataEncoder:
    """Handle all encoding and scaling operations using sklearn"""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.logger = logging.getLogger(__name__)

        # Store fitted encoders/scalers for potential future use
        self.fitted_encoders = {}
        self.fitted_scalers = {}

    def encode_categorical(self, column: str, method: str) -> pd.DataFrame:
        """
        Encode categorical variables using sklearn encoders

        Args:
            column: Name of the column to encode
            method: Encoding method ('one_hot', 'label')
        """
        if column not in self.data.columns:
            self.logger.warning(f"Column {column} not found in dataset")
            return self.data

        try:
            if method == "one_hot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded_data = encoder.fit_transform(self.data[[column]])

                # Create new column names
                feature_names = encoder.get_feature_names_out([column])

                # Replace original column with encoded columns
                self.data.drop(columns=[column], inplace=True)
                for i, name in enumerate(feature_names):
                    self.data[name] = encoded_data[:, i]

                self.fitted_encoders[column] = encoder

            elif method == "label":
                encoder = LabelEncoder()
                self.data[column] = encoder.fit_transform(self.data[column])
                self.fitted_encoders[column] = encoder

            else:
                self.logger.warning(
                    f"Unknown encoding method: {method}. Supported methods: 'one_hot', 'label'"
                )

        except Exception as e:
            self.logger.error(f"Error encoding column {column}: {str(e)}")

        return self.data

    def scale_features(self, column: str, method: str) -> pd.DataFrame:
        """
        Scale numerical features using sklearn scalers

        Args:
            column: Name of the column to scale
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        if column not in self.data.columns:
            self.logger.warning(f"Column {column} not found in dataset")
            return self.data

        try:
            scaler = None
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                self.logger.warning(
                    f"Unknown scaling method: {method}. Supported methods: 'standard', 'minmax', 'robust'"
                )
                return self.data

            # Reshape and scale the data
            self.data[column] = scaler.fit_transform(self.data[[column]])
            self.fitted_scalers[column] = scaler

        except Exception as e:
            self.logger.error(f"Error scaling column {column}: {str(e)}")

        return self.data
