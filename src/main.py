from utils.logger import get_logger
from .pipeline.eda_pipeline import EDAPipeline
from .pipeline.training_pipeline import TrainingPipeline


logger = get_logger(__name__)


def main():
    # ... setup logging and sample data creation ...

    # EDA Pipeline Configuration (minimal preprocessing)
    eda_config = {
        "drop_columns": [],  # Only drop columns that are completely irrelevant
        "na_threshold": None,  # Don't drop columns based on NA threshold during EDA
        "median_imputation": [],  # No imputation during EDA
        "mean_imputation": [],
        "mode_imputation": [],
        "outlier_columns": {},  # No outlier removal during EDA
    }

    # Full Training Pipeline Configuration
    preprocess_config = {
        "drop_columns": [],
        "na_threshold": 0.3,
        "median_imputation": ["age", "experience"],
        "mean_imputation": ["salary"],
        "mode_imputation": ["department"],
        "outlier_columns": {
            "salary": {"method": "iqr", "threshold": 1.5},
            "age": {"method": "zscore", "threshold": 3},
        },
    }

    feature_config = {
        "interactions": [
            {"columns": ["age", "experience"], "name": "age_experience_interaction"}
        ],
        "ratios": [
            {
                "numerator": "salary",
                "denominator": "experience",
                "name": "salary_per_year_experience",
            }
        ],
    }

    encoder_config = {
        "categorical_encoding": {"department": "one_hot", "education": "label"},
        "feature_scaling": {
            "salary": "standard",
            "age": "minmax",
            "salary_per_year_experience": "standard",
            "age_experience_interaction": "robust",
        },
    }

    # Run EDA Pipeline
    eda_pipeline = EDAPipeline()
    eda_data = eda_pipeline.run(
        file_path="sample_data.csv",
        basic_config=eda_config,
    )

    # Run Training Pipeline
    training_pipeline = TrainingPipeline()
    final_data = training_pipeline.run(
        file_path="sample_data.csv",
        preprocess_config=preprocess_config,
        feature_config=feature_config,
        encoder_config=encoder_config,
    )
