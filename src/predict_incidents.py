import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Load data
data = pd.read_csv('../data/NHTSA_crash_data.csv')
print(data.head())    # Quick preview
print(data.columns)   # See available columns for prediction



# Find injury-related columns
injury_cols = [col for col in data.columns if 'Injury' in col or 'injury' in col]
if injury_cols:
    target_col = injury_cols[0]
    df = data.dropna(subset=[target_col])
    X = df.select_dtypes(include=['number']).drop(columns=[target_col], errors='ignore')
    # Encode some categorical features
    for col in ['Make', 'Model', 'Source_System', 'Driver Operator Type']:
        if col in df:
            X[col] = df[col].astype('category').cat.codes
    y = df[target_col]
else:
    print('No injury-related column found for prediction.')
    exit()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
