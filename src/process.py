import pandas as pd

def process(path: str = '../data/history.csv'):
    df = pd.read_csv(path)
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'])
    df['tenure'] = df['tenure'].fillna(df['TotalCharges']//df['MonthlyCharges'])
    
    df.dropna(inplace=True)

    df['tenureCategory'] = df['tenure'].apply(
        lambda x:
        '0-11' if x < 12
        else '12-23' if x < 24
        else '24-35' if x < 36
        else '36-47' if x < 48
        else '48+'
    )
    df['MonthlyChargesCategory'] = df['MonthlyCharges'].apply(
        lambda x:
        '0-40' if x <= 40
        else '41-80' if x <= 80
        else '80+'
    )

    return df