from zenml import step
import pandas as pd
import logging
import mlflow
class IngestData:
    def __init__(self,data_path:str):
        self.data_path=data_path
    def get_data(self):
        '''
        Ingesting the data from data_path
        Returns: the ingested data.
        '''
        logging.info(f"Ingesting data from {self.data_path}")
        data=pd.read_csv(self.data_path)
        return data

@step
def ingest_df( data_path:str ) -> pd.DataFrame:
    '''
    Ingesting data from the datapath,
    Args: 
        data path: path to the data.
        
    Returns:
        pd.Dataframe:the ingested data. 
    '''
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        logging.info("Ingesting data completed.")
        return df
    except Exception as e:
        logging.error("Error while ingesting data")
        raise e            
        
    