import sys
import os

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object


from dataclasses import dataclass

from src.components.model_trainer import ModelTrainer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    label_encoder_file_path: str = os.path.join("artifacts", "label_encoder.pkl")


class DataTransformation:
    def __init__(self):
        self.target_column = "churn"
        self.label_encoder = LabelEncoder()
        self.data_transformation_config = DataTransformationConfig()
        self.numerical_columns = [
                                    "tenure_months",
                                    "monthly_logins",
                                    "customer_satisfaction_score",
                                    "years_in_business",
                                    "monthly_subscription_fee_gbp",
                                    "discount_percentage",
                                    "employee_count",
                                    "active_users",
                                    "feature_usage_score",
                                    "avg_session_duration_minutes",
                                    "projects_created",
                                    "integrations_enabled",
                                    "support_tickets_last_6m",
                                    "avg_resolution_time_hours",
                                    "payment_failures_last_12m",
                                    "email_open_rate",
                                    "webinar_attendance_count",
                                    "monthly_recurring_revenue",
                                    "lifetime_value"
                                ]
        self.categorical_columns = [
                                    "region",
                                    "subscription_plan",
                                    "contract_type",
                                    "industry",
                                    "company_size",
                                    "customer_segment",
                                    "acquisition_channel"
                                    ]

    def get_data_transformer_object(self):
        try:

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown="ignore")),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {self.categorical_columns}")
            logging.info(f"Numerical columns: {self.numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, self.numerical_columns),
                    ('cat_pipeline', cat_pipeline, self.categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    

    def initiate_data_transformation(self, validated_train_path, validated_test_path):

        try:
            train_df = pd.read_csv(validated_train_path)
            test_df = pd.read_csv(validated_test_path)

            logging.info(f"Training dataset shape: {train_df.shape}")

            logging.info(f"Testing dataset shape: {test_df.shape}")

            logging.info("Creating preprocessing pipeline.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = self.target_column
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Encoding target column.")

            target_feature_train_df = self.label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = self.label_encoder.transform(target_feature_test_df)

            logging.info(f"Applying preprocessing pipeline.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saving preprocessing and label encoder objects.")


            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            save_object(
                file_path=self.data_transformation_config.label_encoder_file_path,
                obj=self.label_encoder
            )
            logging.info("Artifacts saved successfully.")

            return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                    self.data_transformation_config.label_encoder_file_path
                )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":

    from src.components.data_transformation import DataTransformation

    data_transformation = DataTransformation()

    train_arr, test_arr, _, _ = data_transformation.initiate_data_transformation(
        validated_train_path="artifacts/validated_train.csv",
        validated_test_path="artifacts/validated_test.csv"
    )

    model_trainer = ModelTrainer()

    results = model_trainer.initiate_model_trainer(
        train_array=train_arr,
        test_array=test_arr
    )

    print("\nFinal Results")
    print(results)