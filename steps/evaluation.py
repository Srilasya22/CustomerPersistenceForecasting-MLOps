import logging
from zenml import step
import pandas as pd
from src.evaluation import MSE,R2,RMSE,F1,Recall,ROC,Precision,Accuracy
from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
from zenml.client import Client
import mlflow


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
    Annotated[float, "roc"],
    Annotated[float, "f1"],
    Annotated[float, "recall"],
    Annotated[float, "precision"],
    Annotated[float, "accuracy"]
]:
    # Rest of the code remains the same
    # ...

    '''
    Evaluating the model.
    
    Args:
        The testing data.
    '''
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("MSE",mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("R2",r2)
    
        roc_class = ROC()
        roc= roc_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("ROC",roc)

        f1_class = F1()
        f1 = f1_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("f1",f1)

        recall_class = Recall()
        recall = recall_class.calculate_scores(y_test,prediction)
       
        mlflow.log_metric("Recall",recall)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("rmse",rmse)

        e_class = Precision()
        precision = e_class.calculate_scores(y_test,prediction)
        # mlflow.log_metric("Precision",precision)
        mlflow.log_metric("Precision",precision)

        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("Accuracy",accuracy)
        return r2,rmse,roc,f1,recall,precision,accuracy
    except Exception as e:
        logging.error("Error while evaluating data {}".format(e))
        raise e
    finally:
        mlflow.end_run()