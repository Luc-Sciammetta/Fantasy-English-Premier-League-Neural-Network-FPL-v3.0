import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error

df = pd.read_csv('predictFuturePoints/fpl_training_data_with_selected.csv')
print("loaded ", len(df), "rows")

target = 'total_points_plus_7'
drop_cols = [target] 

train_df = df[df['season'].isin([2223, 2324])]
test_df = df[df['season'].isin([2425, 2526])]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target]
X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target]

print("Train length ", len(X_train), "Test length ", len(X_test))
print(f"Target stats:\n{y_train.describe()}")

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators = 1000,
    max_depth = 3, 
    learning_rate = 0.01,
    random_state = 42,
    eval_metric='mae',
    early_stopping_rounds = 20,
)

model.fit(
    X_train,
    y_train,
    eval_set = [(X_test, y_test)], 
    verbose = 20,
)

y_pred = model.predict(X_test)

print(f"\nMAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Mean predicted: {y_pred.mean():.4f}")
print(f"Mean actual: {y_test.mean():.4f}")
print(f"Total predicted points: {y_pred.sum():.0f}")
print(f"Total actual points: {y_test.sum()}")

# feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature importance:")
print(importance.to_string(index=False))

model.save_model('predictFuturePoints/future_points_model.json')
print("\nModel saved")