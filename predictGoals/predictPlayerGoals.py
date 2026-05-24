import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error

df = pd.read_csv('predictGoals/goals_training_data.csv')
print("loaded ", len(df), "rows")

target = 'goals_scored'
drop_cols = ['kickoff_time', 'opponent_team_id', target] 

train_df = df[df['season'].isin([2223, 2324])]
test_df = df[df['season'].isin([2425, 2526])]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target]
X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target]

print("Train length ", len(X_train), "Test length ", len(X_test))
print(f"Target stats:\n{y_train.describe()}")

model = xgb.XGBRegressor(
    objective='count:poisson',
    n_estimators = 500,
    max_depth = 5, 
    learning_rate = 0.05,
    random_state = 42,
    eval_metric='poisson-nloglik',
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
print(f"Total predicted goals: {y_pred.sum():.0f}")
print(f"Total actual goals: {y_test.sum()}")

# feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature importance:")
print(importance.to_string(index=False))

model.save_model('predictGoals/goals_model.json')
print("\nModel saved")