import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.logger import logging
from src.exception import CustomException
from src.utils import (evaluate_models,tune_models,save_object)


@dataclass
class ModelTrainerConfig:

    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):

        try:

            logging.info("Starting baseline model evaluation.")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Extra Trees": ExtraTreesClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "XGBoost": XGBClassifier(random_state=42,eval_metric="logloss"),
                "CatBoost": CatBoostClassifier(verbose=False,random_state=42)
            }

            params = {
                 "Random Forest":{
                                    "n_estimators": [100, 200, 300, 500],

                                    "max_depth": [None, 10, 20, 30],

                                    "min_samples_split": [2, 5, 10],

                                    "min_samples_leaf": [1, 2, 4],

                                    "max_features": ["sqrt", "log2"]

                                     },
                "Gradient Boosting": {

                                            "n_estimators": [100, 200, 300],

                                            "learning_rate": [0.01, 0.05, 0.1],

                                            "max_depth": [3, 5, 7],

                                            "subsample": [0.8, 1.0]

                                        },
                "XGBoost": {

                                            "n_estimators": [100, 200, 300],

                                            "learning_rate": [0.01, 0.05, 0.1],

                                            "max_depth": [3, 5, 7],

                                            "subsample": [0.8, 1.0],

                                            "colsample_bytree": [0.8, 1.0]

                                        },
                "CatBoost": {

                                        "iterations": [100, 200, 300],

                                        "depth": [4, 6, 8],

                                        "learning_rate": [0.01, 0.05, 0.1],

                                        "l2_leaf_reg": [1, 3, 5]

                                    }
                
            }

            results = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            logging.info("Baseline evaluation completed successfully.")

            model_report = results["metrics"]
            trained_models = results["trained_models"]

            sorted_models = sorted(
                model_report.items(),
                key=lambda x: x[1]["f1_score"],
                reverse=True
            )

            logging.info("=" * 95)
            logging.info("BASELINE MODEL PERFORMANCE")
            logging.info("=" * 95)

            print("\n")
            print("=" * 95)
            print(
                f"{'Model':25}"
                f"{'Accuracy':>12}"
                f"{'Precision':>12}"
                f"{'Recall':>12}"
                f"{'F1':>12}"
                f"{'ROC-AUC':>12}"
            )
            print("=" * 95)

            for model_name, metrics in sorted_models:

                logging.info(
                    f"{model_name:25}"
                    f"{metrics['accuracy']:>12.4f}"
                    f"{metrics['precision']:>12.4f}"
                    f"{metrics['recall']:>12.4f}"
                    f"{metrics['f1_score']:>12.4f}"
                    f"{metrics['roc_auc']:>12.4f}"
                )

                print(
                    f"{model_name:25}"
                    f"{metrics['accuracy']:>12.4f}"
                    f"{metrics['precision']:>12.4f}"
                    f"{metrics['recall']:>12.4f}"
                    f"{metrics['f1_score']:>12.4f}"
                    f"{metrics['roc_auc']:>12.4f}"
                )

            print("=" * 95)
            logging.info("=" * 95)

            logging.info("Starting hyperparameter tuning...")

            tuned_results = tune_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                trained_models=trained_models,
                params=params
            )

            logging.info("Hyperparameter tuning completed successfully.")

            tuned_report = tuned_results["metrics"]
            tuned_models = tuned_results["trained_models"]

            sorted_tuned_models = sorted(
                tuned_report.items(),
                key=lambda x: x[1]["f1_score"],
                reverse=True
            )

            logging.info("=" * 95)
            logging.info("TUNED MODEL PERFORMANCE")
            logging.info("=" * 95)

            print("\n")
            print("=" * 95)
            print("TUNED MODEL PERFORMANCE")
            print("=" * 95)

            print(
                f"{'Model':25}"
                f"{'Accuracy':>12}"
                f"{'Precision':>12}"
                f"{'Recall':>12}"
                f"{'F1':>12}"
                f"{'ROC-AUC':>12}"
            )

            print("=" * 95)

            for model_name, metrics in sorted_tuned_models:

                logging.info(
                    f"{model_name:25}"
                    f"{metrics['accuracy']:>12.4f}"
                    f"{metrics['precision']:>12.4f}"
                    f"{metrics['recall']:>12.4f}"
                    f"{metrics['f1_score']:>12.4f}"
                    f"{metrics['roc_auc']:>12.4f}"
                )

                print(
                    f"{model_name:25}"
                    f"{metrics['accuracy']:>12.4f}"
                    f"{metrics['precision']:>12.4f}"
                    f"{metrics['recall']:>12.4f}"
                    f"{metrics['f1_score']:>12.4f}"
                    f"{metrics['roc_auc']:>12.4f}"
                )

            print("=" * 95)
            logging.info("=" * 95)
            best_model_name = sorted_tuned_models[0][0]

            best_model = tuned_models[best_model_name]

            best_metrics = tuned_report[best_model_name]

            logging.info(
                f"Best tuned model selected: {best_model_name}"
            )

            logging.info(f"Saving {best_model_name} model...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                f"Best model saved to: {self.model_trainer_config.trained_model_file_path}"
            )
            
            return {
                "best_model": best_model_name,
                "metrics": best_metrics
                
                }

        except Exception as e:

            raise CustomException(e, sys)