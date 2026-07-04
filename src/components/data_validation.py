import os
import sys

import pandas as pd

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataValidationConfig:

    validated_train_path: str = os.path.join(
        "artifacts",
        "validated_train.csv"
    )

    validated_test_path: str = os.path.join(
        "artifacts",
        "validated_test.csv"
    )


class DataValidation:

    def __init__(self):

        self.validation_config = DataValidationConfig()

        self.categorical_columns = [
            "region",
            "subscription_plan",
            "contract_type",
            "industry",
            "company_size",
            "customer_segment",
            "acquisition_channel"
        ]


    def check_empty_dataset(self, df):

        if df.empty:

            raise CustomException(
                "Dataset is empty.",
                sys
            )

        logging.info("Dataset is not empty.")


    def validate_target(self, df):

        if "churn" not in df.columns:

            raise CustomException(
                "Target column 'churn' not found.",
                sys
            )

        logging.info("Target column found.")


    def remove_duplicates(self, df):

        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:

            df = df.drop_duplicates()

        logging.info(
            f"Removed {duplicate_count} duplicate rows."
        )

        return df


    def standardize_categories(self, df):

        for col in self.categorical_columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.title()
            )

        logging.info(
            "Categorical values standardized."
        )

        return df


    def clean_numeric_columns(self, df):

        invalid_satisfaction = (
            (
                df["customer_satisfaction_score"] < 1
            ) |
            (
                df["customer_satisfaction_score"] > 10
            )
        ).sum()

        df["customer_satisfaction_score"] = (
            df["customer_satisfaction_score"]
            .clip(1, 10)
        )

        logging.info(
            f"Corrected {invalid_satisfaction} invalid satisfaction scores."
        )


        invalid_tenure = (
            df["tenure_months"] < 0
        ).sum()

        df["tenure_months"] = (
            df["tenure_months"]
            .clip(lower=0)
        )

        logging.info(
            f"Corrected {invalid_tenure} negative tenure values."
        )

        return df


    def validate_business_rules(self, df):

        if (df["employee_count"] < 0).any():

            raise CustomException(
                "Negative employee count detected.",
                sys
            )

        if (df["monthly_logins"] < 0).any():

            raise CustomException(
                "Negative monthly logins detected.",
                sys
            )

        if (
            (
                df["email_open_rate"] < 0
            ) |
            (
                df["email_open_rate"] > 100
            )
        ).any():

            raise CustomException(
                "Invalid email open rate detected.",
                sys
            )

        if (
            (
                df["discount_percentage"] < 0
            ) |
            (
                df["discount_percentage"] > 100
            )
        ).any():

            raise CustomException(
                "Invalid discount percentage detected.",
                sys
            )

        if (
            df["active_users"] >
            df["employee_count"]
        ).any():

            raise CustomException(
                "Active users cannot exceed employee count.",
                sys
            )

        if (
            df["monthly_subscription_fee_gbp"] <= 0
        ).any():

            raise CustomException(
                "Invalid subscription fee detected.",
                sys
            )

        if (
            df["lifetime_value"] <= 0
        ).any():

            raise CustomException(
                "Invalid lifetime value detected.",
                sys
            )

        logging.info(
            "Business rule validation completed successfully."
        )


    def initiate_data_validation(
        self,
        train_path,
        test_path
    ):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                "Train and Test datasets loaded."
            )

            for df in [train_df, test_df]:

                self.check_empty_dataset(df)

                self.validate_target(df)

            train_df = self.remove_duplicates(train_df)
            test_df = self.remove_duplicates(test_df)

            train_df = self.standardize_categories(train_df)
            test_df = self.standardize_categories(test_df)

            train_df = self.clean_numeric_columns(train_df)
            test_df = self.clean_numeric_columns(test_df)

            self.validate_business_rules(train_df)
            self.validate_business_rules(test_df)

            train_df.to_csv(
                self.validation_config.validated_train_path,
                index=False
            )

            test_df.to_csv(
                self.validation_config.validated_test_path,
                index=False
            )

            logging.info(
                "Validated datasets saved successfully."
            )

            return (
                self.validation_config.validated_train_path,
                self.validation_config.validated_test_path
            )

        except Exception as e:

            raise CustomException(e, sys)
        
if __name__ == "__main__":

    data_validation = DataValidation()

    validated_train_path, validated_test_path = (
        data_validation.initiate_data_validation(
            train_path="artifacts/train.csv",
            test_path="artifacts/test.csv"
        )
    )

    print("Validation completed successfully.")
    print(validated_train_path)
    print(validated_test_path)