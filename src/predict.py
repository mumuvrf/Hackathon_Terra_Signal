import pandas as pd
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

from process import process

def predict(path: str = '../data/inference.csv'):
    with open('../models/transformer.pkl', 'rb') as f:
        ct = pickle.load(f)

    with open('../models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    df = process(path)
    ds = df.copy()

    drop_cols = ['TotalCharges', 'CustomerFeedback', 'customerID', 'tenure', 'MonthlyCharges', 'MonthlyIncome']

    ds.drop(columns=drop_cols, inplace=True)

    X = ds
    X = ct.fit_transform(X)

    df['Churn'] = model.predict(X)
    df.drop(columns=['MonthlyChargesCategory', 'tenureCategory'], inplace=True)

    df.to_csv('../data/prediction.csv', index=False)

if __name__ == "__main__":
    predict()
