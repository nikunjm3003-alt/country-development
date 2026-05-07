# importing the libraries
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # THE DATASET HAS NO NULL VALUES YOU CAN CHECK OUT THE NOTEBOOK
    # FEATURE ENGINEERING
    df['trade_balance'] = df['exports'] - df['imports']
    df['real_income'] = df['income'] / (1 + df['inflation']/100)

    # THE INCOME TO GDP 
    df['income_to_gdp'] = df['income'] / df['gdpp']


    # dropping income because it will be redundant 
    df.drop('income',inplace = True, axis = 1)

    # THIS IS AN UNSUPERVISED DATA SO WE DO NOT HAVE ANY TARGET VALUE SO SPILTTING HERE
    num_cols = df.select_dtypes(include = np.number).columns.tolist()
    X_train = df[num_cols]

    # SCALING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # IN THE NOTEBOOK I HAVE APPLIED PCA BUT I THINK HERE I WON'T BECAUSE PCA IS USED WHEN THE NUMBER OF COLUMNS IS VERY HIGH , HERE IT IS QUITE MODERATE (IN THE NOTEBOOK I WAS EXPERIMENTING)

    return X_scaled,scaler 


# THIS FUNCTION WILL BE USED LATER WHEN AN INPUT IS ENTERED BY THE USER

def transform_input(input_dict,scaler):
    df = pd.DataFrame([input_dict]) if isinstance(input_dict, dict) else input_dict
    # FEATURE ENGINEERING
    df['trade_balance'] = df['exports'] - df['imports']
    df['real_income'] = df['income'] / (1 + df['inflation']/100)

    # THE INCOME TO GDP 
    df['income_to_gdp'] = df['income'] / df['gdpp']

    # dropping income because it will be redundant 
    df.drop('income',inplace = True, axis = 1)

    # THIS IS AN UNSUPERVISED DATA SO WE DO NOT HAVE ANY TARGET VALUE SO SPILTTING HERE
    num_cols = df.select_dtypes(include = np.number).columns.tolist()
    df = df[num_cols]

    # SCALING
    X_scaled = scaler.transform(df) # do not need to fit in it again
    return X_scaled

# expecrted_cols list not needed because we are not using encoded_columns
