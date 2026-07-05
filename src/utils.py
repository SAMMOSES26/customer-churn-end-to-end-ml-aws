import os
import sys
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import RandomizedSearchCV
from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train,y_train,X_test,y_test,models):
    """
    Train all baseline models and evaluate them.

            Returns
            -------
            dict
                metrics:
                    Dictionary containing evaluation metrics for every model.

                trained_models:
                    Dictionary containing fitted model objects.
    """

    try:

        model_report = {}

        trained_models = {}

        for model_name, model in models.items():

            logging.info(f"Training {model_name}")

            trained_model = model.fit(X_train,y_train)
            trained_models[model_name] = trained_model

            y_pred = trained_model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred,zero_division=0)
            recall = recall_score(y_test,y_pred,zero_division=0)
            f1 = f1_score(y_test,y_pred,zero_division=0)
            
            try:
                roc_auc = roc_auc_score(
                    y_test,
                    y_pred
                    )
            except ValueError:
                roc_auc = 0.0

            logging.info(f"{model_name:<20} | " f"Accuracy={accuracy:.4f} | " f"Precision={precision:.4f} | "
                            f"Recall={recall:.4f} | " f"F1={f1:.4f} | " f"ROC_AUC={roc_auc:.4f}"
                        )

            model_report[model_name] = {
                                        "accuracy": accuracy,
                                        "precision": precision,
                                        "recall": recall,
                                        "f1_score": f1,
                                        "roc_auc": roc_auc
                                    }

        return {
                    "metrics": model_report,
                    "trained_models": trained_models
                }

    except Exception as e:

        raise CustomException(e, sys)


def tune_models(
    X_train,
    y_train,
    X_test,
    y_test,
    trained_models,
    params
):
    """
    Tune the selected baseline models using RandomizedSearchCV.

    Returns
    -------
    tuned_metrics:
        Dictionary containing evaluation metrics of tuned models.

    tuned_models:
        Dictionary containing tuned model objects.
    """

    try:

        tuned_metrics = {}

        tuned_models = {}

        for model_name, model in trained_models.items():
            if model_name not in params:
                continue

            logging.info(f"Tuning {model_name}...")
            search = RandomizedSearchCV(estimator=model, param_distributions=params[model_name], n_iter=10,
                                        scoring="f1", cv=5, random_state=42, n_jobs=-1)
            
            search.fit(X_train, y_train)

            best_model = search.best_estimator_

            logging.info(f"Best Parameters for {model_name}: {search.best_params_}")
            logging.info(f"Best CV F1 Score: {search.best_score_:.4f}")
            
            tuned_models[model_name] = best_model

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred,zero_division=0)
            recall = recall_score(y_test,y_pred,zero_division=0)
            f1 = f1_score(y_test,y_pred,zero_division=0)

            try:
                roc_auc = roc_auc_score(y_test,y_pred)
            
            except ValueError:
                roc_auc = 0.0

            logging.info(f"{model_name} Tuned | " f"Accuracy={accuracy:.4f} | " f"Precision={precision:.4f} | "
                         f"Recall={recall:.4f} | " f"F1={f1:.4f} | " f"ROC_AUC={roc_auc:.4f}")
            tuned_metrics[model_name] = {"accuracy": accuracy,
                                         "precision": precision,
                                         "recall": recall,
                                         "f1_score": f1,
                                         "roc_auc": roc_auc
                                         }


        return {"metrics": tuned_metrics,"trained_models": tuned_models}
    
    except Exception as e:

        raise CustomException(e, sys)