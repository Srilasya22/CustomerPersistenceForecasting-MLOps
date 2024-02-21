
import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

class Model(ABC):
    """
    Abstract class for machine learning models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model.

        Args:
            X_train : training data
            y_train : training labels
        """
        pass


# Concrete class for Logistic Regression model
class LogisticRegressionModel(Model):
    """Logistic regression model."""

    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X_train, y_train, **kwargs):
        """Training the logistic regression model."""
        try:
            log_reg = LogisticRegression(**kwargs)
            log_reg.fit(X_train, y_train)
            logging.info("Logistic Regression Model training completed")
            return log_reg
        except Exception as e:
            logging.error("Error in training Logistic Regression model: {}".format(e))
            raise e

# Concrete class for Random Forest Classifier model
class RandomForestModel(Model):
    """Random Forest Classifier model."""

    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X_train, y_train, **kwargs):
        """Training the Random Forest Classifier model."""
        try:
            rf = RandomForestClassifier(**kwargs)
            rf.fit(X_train, y_train)
            logging.info("Random Forest Classifier Model training completed")
            return rf
        except Exception as e:
            logging.error("Error in training Random Forest Classifier model: {}".format(e))
            raise e

# Concrete class for Gradient Boosting Classifier model
class GradientBoostingModel(Model):
    """Gradient Boosting Classifier model."""

    def __init__(self):
        self.model = GradientBoostingClassifier()

    def train(self, X_train, y_train, **kwargs):
        """Training the Gradient Boosting Classifier model."""
        try:
            gb = GradientBoostingClassifier(**kwargs)
            gb.fit(X_train, y_train)
            logging.info("Gradient Boosting Classifier Model training completed")
            return gb
        except Exception as e:
            logging.error("Error in training Gradient Boosting Classifier model: {}".format(e))
            raise e

# Concrete class for Support Vector Machine model
class SVMModel(Model):
    """Support Vector Machine model."""

    def __init__(self):
        self.model = SVC()

    def train(self, X_train, y_train, **kwargs):
        """Training the Support Vector Machine model."""
        try:
            svc = SVC(**kwargs)
            svc.fit(X_train, y_train)
            logging.info("Support Vector Machine Model training completed")
            return svc
        except Exception as e:
            logging.error("Error in training Support Vector Machine model: {}".format(e))
            raise e

# Concrete class for K-Nearest Neighbors model
class KNNModel(Model):
    """K-Nearest Neighbors model."""

    def __init__(self):
        self.model = KNeighborsClassifier()

    def train(self, X_train, y_train, **kwargs):
        """Training the K-Nearest Neighbors model."""
        try:
            knn = KNeighborsClassifier(**kwargs)
            knn.fit(X_train, y_train)
            logging.info("K-Nearest Neighbors Model training completed")
            return knn
        except Exception as e:
            logging.error("Error in training K-Nearest Neighbors model: {}".format(e))
            raise e

# Concrete class for XGBoost Classifier model
class XGBoostModel(Model):
    """XGBoost Classifier model."""

    def __init__(self):
        self.model = XGBClassifier()

    def train(self, X_train, y_train, **kwargs):
        """Training the XGBoost Classifier model."""
        try:
            xgb = XGBClassifier(**kwargs)
            xgb.fit(X_train, y_train)
            logging.info("XGBoost Classifier Model training completed")
            return xgb
        except Exception as e:
            logging.error("Error in training XGBoost Classifier model: {}".format(e))
            raise e

# Concrete class for Decision Tree Classifier model
class DecisionTreeModel(Model):
    """Decision Tree Classifier model."""

    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train, **kwargs):
        """Training the Decision Tree Classifier model."""
        try:
            dt = DecisionTreeClassifier(**kwargs)
            dt.fit(X_train, y_train)
            logging.info("Decision Tree Classifier Model training completed")
            return dt
        except Exception as e:
            logging.error("Error in training Decision Tree Classifier model: {}".format(e))
            raise e

