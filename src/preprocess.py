import pandas as pd

def preprocess_nhtsa(df):
    selected = df[['CRASH_SEVERITY', 'WEATHER_CONDITION', 'LIGHT_CONDITION', 'AUTONOMOUS_STATUS']]
    encoded = pd.get_dummies(selected, drop_first=True)
    X = encoded.drop('CRASH_SEVERITY', axis=1)
    y = encoded['CRASH_SEVERITY']
    return X, y
