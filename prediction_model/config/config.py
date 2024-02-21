import pathlib
import os
import prediction_model
PACKAGE_ROOT=pathlib.Path(prediction_model.__file__).resolve().parent #Path of prediction model
DATAPATH=os.path.join(PACKAGE_ROOT,"datasets")
SAVEDMODELPATH=os.path.join(PACKAGE_ROOT,"saved_model")
model_name="LogisticRegression"