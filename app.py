from flask import Flask, request, render_template
from src.exception import CustomException
import sys
from src.pipeline.predict_pipeline import (CustomData,PredictPipeline)


application = Flask(__name__)
app = application

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("home.html")
    else:
            try:

                data = CustomData(

                    region=request.form.get("region"),
                    subscription_plan=request.form.get("subscription_plan"),
                    contract_type=request.form.get("contract_type"),
                    industry=request.form.get("industry"),
                    company_size=request.form.get("company_size"),
                    customer_segment=request.form.get("customer_segment"),
                    acquisition_channel=request.form.get("acquisition_channel"),

                    tenure_months=int(request.form.get("tenure_months")),
                    monthly_logins=float(request.form.get("monthly_logins")),
                    customer_satisfaction_score=float(request.form.get("customer_satisfaction_score")),
                    years_in_business=int(request.form.get("years_in_business")),
                    monthly_subscription_fee_gbp=float(request.form.get("monthly_subscription_fee_gbp")),
                    discount_percentage=float(request.form.get("discount_percentage")),
                    employee_count=int(request.form.get("employee_count")),
                    active_users=int(request.form.get("active_users")),
                    feature_usage_score=float(request.form.get("feature_usage_score")),
                    avg_session_duration_minutes=float(request.form.get("avg_session_duration_minutes")),
                    projects_created=int(request.form.get("projects_created")),
                    integrations_enabled=int(request.form.get("integrations_enabled")),
                    support_tickets_last_6m=int(request.form.get("support_tickets_last_6m")),
                    avg_resolution_time_hours=float(request.form.get("avg_resolution_time_hours")),
                    payment_failures_last_12m=int(request.form.get("payment_failures_last_12m")),
                    email_open_rate=float(request.form.get("email_open_rate")),
                    webinar_attendance_count=int(request.form.get("webinar_attendance_count")),
                    monthly_recurring_revenue=float(request.form.get("monthly_recurring_revenue")),
                    lifetime_value=float(request.form.get("lifetime_value"))

                )
                pred_df = data.get_data_as_dataframe()
                predict_pipeline = PredictPipeline()
                prediction, probability = predict_pipeline.predict(pred_df)
                if prediction == "Yes":
                    prediction_message = "⚠ High Risk of Customer Churn"
                else:
                    prediction_message = "✅ Customer Likely to Stay"
                return render_template("home.html", prediction=prediction_message, confidence=confidence)

            except Exception as e:
                raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(debug=True)