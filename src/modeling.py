import lightgbm as lgb

def train_model(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_set = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'multiclass',
        'num_class': y.nunique(),
        'metric': 'multi_logloss'
    }
    model = lgb.train(params, train_set, num_boost_round=100)
    return model
