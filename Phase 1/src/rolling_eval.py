import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline



# Perform rolling evaluation

rolling_results = []
seasons = ['2017-18', '2018-19', '2019-20', '2020-21']
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}


for i in range(1, len(seasons)):
    train_seasons = seasons[:i]
    test_season = seasons[i]
    
    X_train = pd.concat([pd.read_csv(f'../data/processed/features_{season}.csv').drop(['FTR', 'Date', 'HomeTeam', 'AwayTeam', 'Div'], axis=1) for season in train_seasons])
    y_train = pd.concat([pd.read_csv(f'../data/processed/features_{season}.csv')['FTR'] for season in train_seasons])
    
    X_test = pd.read_csv(f'../data/processed/features_{test_season}.csv').drop(['FTR', 'Date', 'HomeTeam', 'AwayTeam', 'Div'], axis=1)
    y_test = pd.read_csv(f'../data/processed/features_{test_season}.csv')['FTR']
    
    # Baseline model: predict home win for all matches
    y_pred_baseline = ['H'] * len(y_test)
    report = classification_report(y_test, y_pred_baseline, output_dict=True)
    rolling_results.append({
        'Test Season': test_season,
        'Model': 'Baseline (Home Win)',
        'Accuracy': report['accuracy'],
        'Precision_H': report['H']['precision'],
        'Recall_H': report['H']['recall'],
        'F1_H': report['H']['f1-score'],
        'Precision_D': report['D']['precision'],
        'Recall_D': report['D']['recall'],
        'F1_D': report['D']['f1-score'],
        'Precision_A': report['A']['precision'],
        'Recall_A': report['A']['recall'],
        'F1_A': report['A']['f1-score']
    })
    # predict most frequent outcome in training data
    most_frequent_outcome = y_train.mode()[0]
    y_pred_most_frequent = [most_frequent_outcome] * len(y_test)
    report = classification_report(y_test, y_pred_most_frequent, output_dict=True)
    rolling_results.append({
        'Test Season': test_season,
        'Model': 'Most Frequent Outcome',
        'Accuracy': report['accuracy'],
        'Precision_H': report['H']['precision'],
        'Recall_H': report['H']['recall'],
        'F1_H': report['H']['f1-score'],
        'Precision_D': report['D']['precision'],
        'Recall_D': report['D']['recall'],
        'F1_D': report['D']['f1-score'],
        'Precision_A': report['A']['precision'],
        'Recall_A': report['A']['recall'],
        'F1_A': report['A']['f1-score']
    })

    
    for name, model in models.items():
        if name == 'Logistic Regression':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000))
            ])
            X_train_scaled = model.named_steps['scaler'].fit_transform(X_train)
            X_test_scaled = model.named_steps['scaler'].transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        
        # map indices to probabilities for log loss and brier score
        classes = model.classes_
        class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        P_home_win = y_proba[:, class_to_index['H']]
        P_draw = y_proba[:, class_to_index['D']]
        P_away_win = y_proba[:, class_to_index['A']]

        # calculate log loss and brier score
        log_loss_total = 0
        brier_score_total = 0
        for i in range(len(y_test)):
            if y_test.iloc[i] == 'H':
                prob = P_home_win[i]
                brier_score_total += (np.square(P_home_win[i] - 1) + np.square(P_draw[i]) + np.square(P_away_win[i]))
            elif y_test.iloc[i] == 'D':
                prob = P_draw[i]
                brier_score_total += (np.square(P_home_win[i]) + np.square(P_draw[i] - 1) + np.square(P_away_win[i]))
            elif y_test.iloc[i] == 'A':
                prob = P_away_win[i]
                brier_score_total += (np.square(P_home_win[i]) + np.square(P_draw[i]) + np.square(P_away_win[i] - 1))
            log_loss_total += -np.log(max(prob, 1e-15)) 

        
        log_loss_avg = log_loss_total / len(y_test)
        brier_score_avg = brier_score_total / len(y_test)

     
        report = classification_report(y_test, y_pred, output_dict=True)
        rolling_results.append({
            'Test Season': test_season,
            'Model': name,
        'Accuracy': report['accuracy'],
        'Log Loss': log_loss_avg,
        'Brier Score': brier_score_avg,
        'Precision_H': report['H']['precision'],
        'Recall_H': report['H']['recall'],
        'F1_H': report['H']['f1-score'],
        'Precision_D': report['D']['precision'],
        'Recall_D': report['D']['recall'],
        'F1_D': report['D']['f1-score'],
        'Precision_A': report['A']['precision'],
        'Recall_A': report['A']['recall'],
        'F1_A': report['A']['f1-score']
    })

rolling_results_df = pd.DataFrame(rolling_results)

# Calculate performance metrics across all test seasons for each model
final_results = rolling_results_df.groupby('Model').agg({
    'Accuracy': 'mean',
    'Log Loss': 'mean', 
    'Brier Score': 'mean', 
    'Precision_H': 'mean',
    'Recall_H': 'mean',
    'F1_H': 'mean',
    'Precision_D': 'mean',
    'Recall_D': 'mean',
    'F1_D': 'mean',
    'Precision_A': 'mean',
    'Recall_A': 'mean',
    'F1_A': 'mean'
}).reset_index()

# calculate standard deviation of accuracy for each model
final_results['Accuracy_std'] = rolling_results_df.groupby('Model')['Accuracy'].std().values
rolling_results_df.to_csv('../data/processed/rolling_evaluation_results.csv', index=False)
final_results.to_csv('../data/processed/rolling_evaluation_summary.csv', index=False)

