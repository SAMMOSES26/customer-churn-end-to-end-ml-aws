import os
import sys
import pandas as pd
import pickle

from src.exception import CustomException


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:
            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            label_encoder_path = os.path.join(
                "artifacts",
                "label_encoder.pkl"
            )


            with open(preprocessor_path, "rb") as file:
                preprocessor = pickle.load(file)

            with open(model_path, "rb") as file:
                model = pickle.load(file)

            with open(label_encoder_path, "rb") as file:
                label_encoder = pickle.load(file)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            probability = model.predict_proba(data_scaled)
            prediction = label_encoder.inverse_transform(prediction)

            return prediction[0], probability



        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,

        region: str,
        subscription_plan: str,
        contract_type: str,
        industry: str,
        company_size: str,
        customer_segment: str,
        acquisition_channel: str,

        tenure_months: int,
        monthly_logins: int,
        customer_satisfaction_score: float,
        years_in_business: int,
        monthly_subscription_fee_gbp: float,
        discount_percentage: float,
        employee_count: int,
        active_users: int,
        feature_usage_score: float,
        avg_session_duration_minutes: float,
        projects_created: int,
        integrations_enabled: int,
        support_tickets_last_6m: int,
        avg_resolution_time_hours: float,
        payment_failures_last_12m: int,
        email_open_rate: float,
        webinar_attendance_count: int,
        monthly_recurring_revenue: float,
        lifetime_value: float
    ):
    
                self.region = region
                self.subscription_plan = subscription_plan
                self.contract_type = contract_type
                self.industry = industry
                self.company_size = company_size
                self.customer_segment = customer_segment
                self.acquisition_channel = acquisition_channel

                self.tenure_months = tenure_months
                self.monthly_logins = monthly_logins
                self.customer_satisfaction_score = customer_satisfaction_score
                self.years_in_business = years_in_business
                self.monthly_subscription_fee_gbp = monthly_subscription_fee_gbp
                self.discount_percentage = discount_percentage
                self.employee_count = employee_count
                self.active_users = active_users
                self.feature_usage_score = feature_usage_score
                self.avg_session_duration_minutes = avg_session_duration_minutes
                self.projects_created = projects_created
                self.integrations_enabled = integrations_enabled
                self.support_tickets_last_6m = support_tickets_last_6m
                self.avg_resolution_time_hours = avg_resolution_time_hours
                self.payment_failures_last_12m = payment_failures_last_12m
                self.email_open_rate = email_open_rate
                self.webinar_attendance_count = webinar_attendance_count
                self.monthly_recurring_revenue = monthly_recurring_revenue
                self.lifetime_value = lifetime_value


    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {

                "region": [self.region],
                "subscription_plan": [self.subscription_plan],
                "contract_type": [self.contract_type],
                "industry": [self.industry],
                "company_size": [self.company_size],
                "customer_segment": [self.customer_segment],
                "acquisition_channel": [self.acquisition_channel],

                "tenure_months": [self.tenure_months],
                "monthly_logins": [self.monthly_logins],
                "customer_satisfaction_score": [self.customer_satisfaction_score],
                "years_in_business": [self.years_in_business],
                "monthly_subscription_fee_gbp": [self.monthly_subscription_fee_gbp],
                "discount_percentage": [self.discount_percentage],
                "employee_count": [self.employee_count],
                "active_users": [self.active_users],
                "feature_usage_score": [self.feature_usage_score],
                "avg_session_duration_minutes": [self.avg_session_duration_minutes],
                "projects_created": [self.projects_created],
                "integrations_enabled": [self.integrations_enabled],
                "support_tickets_last_6m": [self.support_tickets_last_6m],
                "avg_resolution_time_hours": [self.avg_resolution_time_hours],
                "payment_failures_last_12m": [self.payment_failures_last_12m],
                "email_open_rate": [self.email_open_rate],
                "webinar_attendance_count": [self.webinar_attendance_count],
                "monthly_recurring_revenue": [self.monthly_recurring_revenue],
                "lifetime_value": [self.lifetime_value]

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:

            raise CustomException(e, sys)