from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
from prediction_model.config import config
import os
if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    file_path=os.path.join(config.DATAPATH,'data.csv')
    train_pipeline(data_path=file_path)