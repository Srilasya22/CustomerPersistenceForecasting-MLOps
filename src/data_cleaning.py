import logging
from abc import ABC,abstractmethod
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from imblearn.over_sampling import SMOTE
"""
Design Pattern used: Strategy Pattern
Components:-
1. Abstract Class - must do methods in concrete classes
2. Concrete Class(concrete strategies) - 
3. Context Class
4. Main Code
"""

#Abstract class
class DataStrategy(ABC):
    """
    Abstract class defining for handling data.
    """
    @abstractmethod
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Args:
            pd.Dataframe: Inserting data
        Returns:
            pd.Dataframe , series: Either dataframe or series.
        """
        pass
    
    
#Concrete class
class DataPreProcessStrategy (DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            #dropping unnecessary columns 
            unnecessary_columns = ['customer_id'] 
            data=data.drop(columns=unnecessary_columns,axis=1)
            categorical_features = ['country', 'gender']
            numeric_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
            # Define mapping for countries
            country_mapping = {'Spain': 0, 'France': 1, 'Germany': 2}
            # Map country values to integers
            data['country'] = data['country'].map(country_mapping)

            # Define mapping for genders
            gender_mapping = {'Male': 0, 'Female': 1}
            # Map gender values to integers
            data['gender'] = data['gender'].map(gender_mapping)

            
            return data
        except Exception as e:
            logging.error("Error in processing data: {}".format(e))
            raise e
        
        
            
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test

    Args:
        np.Dataframe:
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            X = data.drop('churn', axis=1)
            y = data['churn']
            # Handle imbalance using SMOTE
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
       
       
#context class 
class DataCleaning:
    """
    Context-class : the "context class" is a class that contains the main business logic and has a dependency on one or more strategies.
    class for cleaning data which preprocess the data and divides it into train and test
    """
    def __init__(self, data: pd.DataFrame, strategy:DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling DataCleaning: {}".format(e))
            raise e