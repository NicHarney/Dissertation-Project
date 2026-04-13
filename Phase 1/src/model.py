import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# Load features
features_2017_18 = pd.read_csv('../data/processed/features_2017-18.csv')
features_2018_19 = pd.read_csv('../data/processed/features_2018-19.csv')
features_2019_20 = pd.read_csv('../data/processed/features_2019-20.csv')
features_2020_21 = pd.read_csv('../data/processed/features_2020-21.csv')

#split data chronologically
X_train = pd.concat([features_2017_18.drop(['FTR'], axis=1), features_2018_19.drop(['FTR'], axis=1), features_2019_20.drop(['FTR'], axis=1)])
y_train = pd.concat([features_2017_18['FTR'], features_2018_19['FTR'], features_2019_20['FTR']])

X_test = features_2020_21.drop(['FTR'], axis=1)
y_test = features_2020_21['FTR']

# preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.drop(['Date', 'HomeTeam', 'AwayTeam', 'Div'], axis=1))
X_test_scaled = scaler.transform(X_test.drop(['Date', 'HomeTeam', 'AwayTeam', 'Div'], axis=1))

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

## baseline model predictions

# Baseline model: predict home win for all matches
y_pred_baseline = ['H'] * len(y_test)
print("Baseline Model Performance:")
print(classification_report(y_test, y_pred_baseline))

# predict most frequent outcome in training data
most_frequent_outcome = y_train.mode()[0]
y_pred_most_frequent = [most_frequent_outcome] * len(y_test)
print("Most Frequent Outcome Model Performance:")
print(classification_report(y_test, y_pred_most_frequent))

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"{name} Performance:")
    print(classification_report(y_test, y_pred))


 