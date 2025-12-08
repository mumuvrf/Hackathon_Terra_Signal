import pandas as pd
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from process import process

def train(path: str = '../data/history.csv'):
    ds = process(path)

    drop_cols = ['TotalCharges', 'CustomerFeedback', 'customerID', 'tenure', 'MonthlyCharges', 'MonthlyIncome']
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'tenureCategory', 'MonthlyChargesCategory'
    ]

    ds.drop(columns=drop_cols, inplace=True)

    X = ds.drop(['Churn'], axis=1)
    y = ds['Churn']

    ohe = OneHotEncoder(handle_unknown='ignore', drop='first')  # drop para evitar dummy trap
    ct = ColumnTransformer([
        ('cat', ohe, categorical_cols)
    ], remainder='passthrough')

    X = ct.fit_transform(X)

    y = y.apply(lambda x: 1 if x == 'Yes' else 0)

    clr = LogisticRegression(class_weight='balanced')
    clr.fit(X, y)

    with open('../models/transformer.pkl', 'wb') as f:
        pickle.dump(ct, f)

    with open('../models/model.pkl', 'wb') as f:
        pickle.dump(clr, f)

if __name__ == "__main__":
    train()