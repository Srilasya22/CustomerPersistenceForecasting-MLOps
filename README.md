# Customer Persistence Forecasting-MLOps

Customer persistence forecasting is like predicting whether the customer will churn or not so that advanced machine learning techniques and MLOps strategies can be used to retain the customers.

## About the project

Discovering the secret sauce to keep customers coming back for more! Unleash the power of machine learning and MLOps practices to understand how customers behave, so we can craft irresistible strategies that make them stick around. 

Key Features:
- Data preprocessing and feature engineering
- Development of predictive models for customer persistence.
- Integrating MLOps principles for efficient model deployment

## Data

The data used in this project is from kaggle website
1. ``customer_id``: An identifier for each customer 
2. ``tenure``: The length of time the customer has been with the bank. 
3. ``credit_score``: A numerical value representing the credit worthiness of a customer
4. ``Age``:The age of the customer
5. ``gender``: Gender of the customer 
6. ``Country``: The country where the customer is located
7. ``Balance``: The amount of money in the customer's account
8. ``products_number``: The number of products the customer has with the bank 
9. ``credit_card``: A binary variable indicating whether the customer has a credit card with the bank
10. ``active_member``: A binary variable indicating whether the customer is an active member of the bank
11. ``estimated_salary``: The estimated salary of the customer  
12. `` Churn``: Whether the customer has churned

## Deployment
To set up this project on your local machine, follow these steps:

1. Develop the project in python 3.9 version
2. Clone the repository: `git clone https://github.com/Srilasya22/CustomerPersistenceForecasting-MLOps.git`
3. Install the required dependencies: `pip install -r requirements.txt`

## Exploring Data

![image](https://github.com/Srilasya22/CustomerPersistenceForecasting-MLOps/assets/113256681/1130e7b2-61fb-4bac-a7b6-78656de91cb2)



## ZenML And MLFlow
``ZenML`` is a machine learning operations (MLOps) framework designed to simplify and streamline the end-to-end machine learning workflow. It provides a structured and organized approach to managing machine learning projects by offering features such as experiment tracking, data versioning, pipeline orchestration, and model deployment.

``MLflow`` is an open-source platform that helps manage the end-to-end machine learning lifecycle. It provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models. MLflow consists of several components:
1. ``Tracking``: MLflow Tracking allows you to log parameters, code versions, metrics, and output files when running your machine learning code to understand what works and what doesn't
2.``Projects``: MLflow Projects is a format for packaging data science code in a reusable and reproducible way, making it easy to share and reproduce results
3.``Models``: MLflow Models provides a standardized format for packaging machine learning models that can be used in a variety of downstream tools, such as batch inference and real-time serving
4.``Registry``: MLflow Registry is a centralized repository for managing and versioning machine learning models, allowing collaboration and sharing across teams

####To install Zenml:
- Create and activate virtual environment
```bash
pip install zenml["server"]
zenml up
```

If you are running the `run_deployment.py` script, you will also need to install some integrations using ZenML:

```bash
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack is as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

### Training Pipeline

Our standard training pipeline consists of several steps:
- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data by removing the unwanted columns,converting categorical columns to numerical columns and by performing oversampling as the dataset is highly imbalanced.
- `train_model`: This step will train the model.
- `evaluation`: This step will evaluate the model and save the metrics using MLflow log_metric

```bash
python run_pipeline.py
```
![Screenshot 2024-02-21 184354](https://github.com/Srilasya22/CustomerPersistenceForecasting-MLOps/assets/113256681/a12683d0-44c8-4d0b-8964-605ef3bc50fa)


## Deployment pipeline

We have another pipeline, the `deployment_pipeline.py`, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria.

```bash
python run_deployment.py
```

## Demo Streamlit App

There is a live demo of this project using Streamlit which you can find here. It takes some input features and predicts whether the customer churns or not. If you want to run this Streamlit app in your local system, you can run the following command to access the app locally:
```bash
streamlit run streamlit_app.py
```
![Screenshot 2024-02-22 030449](https://github.com/Srilasya22/CustomerPersistenceForecasting-MLOps/assets/113256681/8b652b72-78b5-4c75-9d70-de17045537f5)


