import logging 
from abc import ABC,abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
class Evaluation(ABC):
    """Abstract class for evaluating model

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Calculate the score for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        pass

class ROC(Evaluation):
    """
    Evaluation strategy that uses roc_auc_score.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating ROC AUC SCORE")
            # Area under the ROC curve
            auc_roc = roc_auc_score(y_true, y_pred)
            logging.info("ROC_AUC_SCORE: {}".format(auc_roc))
            return auc_roc
        except Exception as e:
            logging.error("Error in calculating ROC_AUC-SCORE, {}".format(e))
            raise e


class F1(Evaluation):
    """
    Evaluation strategy that uses F1.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating F1")
            #F1
            f1 = f1_score(y_true, y_pred)
            logging.info("F1: {}".format(f1))
            return f1
        except Exception as e:
            logging.error("Error in calculating F1, {}".format(e))
            raise e
class Recall(Evaluation):
    """
    Evaluation strategy that uses recall.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating recall")
            #Recall
            recall = recall_score(y_true, y_pred)
            logging.info("Recall: {}".format(recall))
            return recall
        except Exception as e:
            logging.error("Error in calculating recall, {}".format(e))
            raise e
        
class Precision(Evaluation):
    """
    Evaluation strategy that uses precision.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Precision")
            # Precision
            precision = precision_score(y_true, y_pred)
            logging.info("Precision: {}".format(precision))
            return precision
        except Exception as e:
            logging.error("Error in calculating Precision, {}".format(e))
            raise e
        

class Accuracy(Evaluation):
    """
    Evaluation strategy that uses accuracy.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating accuracy")
            # Accuracy
            accuracy = accuracy_score(y_true, y_pred)
            logging.info("Accuracy: {}".format(accuracy))
            return accuracy
        except Exception as e:
            logging.error("Error in calculating accuracy, {}".format(e))
            raise e
        
    
class MSE(Evaluation):
    """
    Evaluation strategy that uses mean squared error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_pred=y_pred,y_true=y_true)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE, {}".format(e))
            raise e
        
class R2(Evaluation):
    """
    Evaluating strategy: Rsquare
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculating Rsquare scores

        Args:
            y_true (ndarray): true labels
            y_pred (ndarray): predictions

        Returns:
            float: rmse
        """
        try:
            logging.info("Calculating Rsquare")
            r2 = r2_score(y_pred=y_pred,y_true=y_true)
            logging.info("R2: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2: ".format(r2))
            
            
class RMSE(Evaluation):
    """Calculating RMSE
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculating RMSE scores

        Args:
            y_true (ndarray): true labels
            y_pred (ndarray): predictions

        Returns:
            float: rmse
        """
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_pred=y_pred,y_true=y_true,squared=False))
            logging.info("MSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating MSE, {}".format(e))
            raise e