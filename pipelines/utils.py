import logging
import pandas as pd
from src.data_cleaning import DataCleaning,DataPreProcessStrategy
import os
from prediction_model.config import config

def get_data_for_test():
    try:
        file_path=os.path.join(config.DATAPATH,'data.csv')
        df = pd.read_csv(file_path)
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df,preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["churn"],axis=1,inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e
