import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
import joblib

st.title("Customer Persistence Forecasting")

   
st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer persistence of a bank based on features like salary,age etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer persistence i.e whether the customer will churn or not """
    )
   
 
st.markdown (
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | customer_id| An identifier for each customer | 
    | tenure   | The length of time the customer has been with the bank. |  
    | credit_score| A numerical value representing the credit worthiness of a customer
    | Age | The age of the customer |
    | gender | Gender of the customer (Male: 1, Female: 0). | 
    | Country | The country where the customer is located |
    | Balance | The amount of money in the customer's account. |
    | products_number | The number of products the customer has with the bank |   
    | credit_card | A binary variable indicating whether the customer has a credit card with the bank.|
    | active_member | A binary variable indicating whether the customer is an active member of the bank.|
    | estimated_salary | The estimated salary of the customer | 
    | Churn   | Whether the customer has churned (Yes: 1, No: 0).   |
    """
)
def predict(credit_score,country,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary):
    if gender== 'Male':
         gender=0
    else:
        gender=1
    if country == 'Spain':
        country=0
    elif country== 'France':
        country=1
    else:
        country=2

    if credit_card == 'Have Credit Card':
        credit_card=1
    else:
        credit_card=0
    if active_member== 'Yes':
        active_member=1
    else:
        active_member=0
    credit_score = int(credit_score)
    age = int(age)
    tenure= int(tenure)
    balance = int(balance)
    products_number = int(products_number)
    estimated_salary=int(estimated_salary)
    input=np.array([[credit_score,country,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary]]).astype(np.float64)
    
        
    model = joblib.load('model.pkl')
    # Make prediction
    prediction = model.predict(input)
    prediction= np.round(prediction)
    # Return the predicted value for Churn
    return prediction
   
credit_score = st.number_input('Enter the credit score',min_value = 0)
age = st.number_input('Enter the age',min_value = 18)
tenure = st.number_input('Enter the tenure',min_value = 0)
balance= st.number_input('Enter the balance',min_value = 0)
products_number = st.number_input('Enter the number of products you own',min_value = 0)
estimated_salary= st.number_input('Enter the salary',min_value = 0)
country= st.radio("Country",('Spain','Germany','France'))
credit_card= st.radio("Credit Card",('Have Credit Card','Dont have Credit Card'))
active_member= st.radio("Active member",('Yes','No'))
gender = st.radio("Gender",('Female','MALE'))

if st.button("Predicted Output"): 
    output = predict(credit_score,country,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary)
    if output==0:
        output1="Not Churn"
    else:
        output1="Churn"  
    st.success("The customer is: {}".format(output1)) 
        
