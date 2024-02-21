import logging
import pandas as pd
from zenml import step 
from src.model_dev import LogisticRegressionModel, RandomForestModel, GradientBoostingModel, SVMModel, KNNModel, XGBoostModel, DecisionTreeModel
from prediction_model.config import config
from sklearn.base import ClassifierMixin
import mlflow
from zenml.client import Client
import pickle
experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> ClassifierMixin:
    """
    Train the data

    Args:
        X_train (pd.DataFrame): The features for training
        y_train (pd.DataFrame): The target variable for training

    Returns:
        trained_model (RegressorMixin): The trained model
    """
    try:
        model = None
        if config.model_name == "KNN":
            mlflow.sklearn.autolog()
            model = KNNModel()
        elif config.model_name == "LogisticRegression":
            mlflow.sklearn.autolog()
            model = LogisticRegressionModel()
        elif config.model_name == "RandomForest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "GradientBoosting":
            mlflow.sklearn.autolog()
            model = GradientBoostingModel()
        elif config.model_name == "SVM":
            mlflow.sklearn.autolog()
            model = SVMModel()
        elif config.model_name == "XGBoost":
            mlflow.sklearn.autolog()
            model = XGBoostModel()
        elif config.model_name == "DecisionTree":
            mlflow.sklearn.autolog()
            model = DecisionTreeModel()
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
        
        # Train the model
        trained_model = model.train(X_train, y_train)
        with open('model.pkl', 'wb') as f:
            pickle.dump(trained_model, f)
        logging.info("Model training completed.")
        return trained_model
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e
    finally:
        mlflow.end_run()
