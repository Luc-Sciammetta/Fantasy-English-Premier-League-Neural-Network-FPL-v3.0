import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

#load the data
df = pd.read_csv('predictMinutes/minutes_training_data.csv')
print("loaded ", len(df), "rows")

target = 'did_player_play'
drop_cols = ['kickoff_time', target] #we dont need these in the features

#split data by season instead of randomly, to prevent leaking future info
train_df = df[df['season'].isin([2223, 2324])]
test_df = df[df['season'].isin([2425, 2526])]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df[target]
X_test = test_df.drop(columns=drop_cols)
y_test = test_df[target]

print("Train length ", len(X_train), "Test length ", len(X_test))
print("Class distribution in train: ", y_train.value_counts(normalize=True))

#create and train the decision tree model(s)
model = xgb.XGBClassifier(
    objective = 'binary:logistic', 
    n_estimators = 500, #number of trees
    max_depth = 5, #how deep each tree can go
    learning_rate = 0.05, #how much each tree contributes to the final prediction
    random_state = 42,
    eval_metric = 'logloss',
    early_stopping_rounds = 20 #stops if the model doesnt improve for 20 trees
)

model.fit( 
    X_train, 
    y_train, 
    eval_set=[(X_test, y_test)],
    verbose = 20, #prints the progress every 20 trees
)

#evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
target_names = ["Did Not Play", "Played < 60 mins", "Played 60+ mins"]

print("\nConfusion Matrix:")
print("Rows = Actual, Columns = Predicted")
print(confusion_matrix(y_test, y_pred))

#see which features mattered the most
importance = pd.DataFrame({
    'feature': X_train.columns, 
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importances:")
print(importance)

#save the model
model.save_model('predictMinutes/minutes_model.json')
print("model saved!")


#when predicting, we want the probabilities, so use predict_proba instead of predict.
