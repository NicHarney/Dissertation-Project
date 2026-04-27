
def run():
    print("Creating graphs...")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from collections import Counter

    df = pd.read_csv('../data/processed/rolling_evaluation_results.csv')

    df = df[pd.notna(df['Log Loss'])]  # Remove rows with NaN log loss

    df = df.groupby('Model').agg({
        'Log Loss': ['mean', 'std'],
        'Brier Score': ['mean', 'std']
    }).reset_index()

    df.columns = ['Model', 'Log Loss', 'Log Loss Std', 'Brier Score', 'Brier Score Std']

    df = df.sort_values('Log Loss')

    # Plotting Log Loss
    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df['Log Loss'], yerr=df['Log Loss Std'],capsize=5, color='skyblue')

    plt.title(f'Log Loss comparison of Machine Learning models (Phase 1)')
    plt.xlabel('Model')
    plt.ylabel('Log Loss')
    plt.ylim(0, max((df['Log Loss']) + df['Log Loss Std']) * 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plotting Brier Score
    df = df.sort_values('Brier Score')
    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df['Brier Score'], yerr=df['Brier Score Std'],capsize=5, color='salmon')
    plt.title(f'Brier Score comparison of Machine Learning models (Phase 1)')
    plt.xlabel('Model')
    plt.ylabel('Brier Score')
    plt.ylim(0, max((df['Brier Score']) + df['Brier Score Std']) * 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # Plot small multiples of home win, draw and away win for actual results each season
    seasons = ['2017-18', '2018-19', '2019-20', '2020-21']
    
    classes = ['H', 'D', 'A']
    x = np.arange(len(seasons))
    width = 0.2
    h_values = []
    d_values = []
    a_values = []
    fig, axes = plt.subplots(figsize=(10, 6), sharey=True)
    for season in seasons:
        season_results = pd.read_csv(f'../data/processed/features_{season}.csv')['FTR']

        counts = Counter(season_results)
        
        h_values.append(counts['H'])
        d_values.append(counts['D'])
        a_values.append(counts['A'])

    plt.bar(x - width, h_values, width, label='Home Win', color='lightblue')
    plt.bar(x, d_values, width, label='Draw', color='lightgreen')
    plt.bar(x + width, a_values, width, label='Away Win', color='red')
  
    plt.xticks(x, seasons)
    plt.xlabel('Season')
    plt.ylabel('Count')
    plt.title('Distribution of Actual Match Outcomes by Season')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Plot small multiples of home win, draw and away win for predicted results each season
    all_seasons_h = np.array(h_values).sum() - h_values[0]
    all_seasons_d = np.array(d_values).sum() - d_values[0]
    all_seasons_a = np.array(a_values).sum() - a_values[0]
    h_values = []
    d_values = []
    a_values = []
    log_loss_values = []
    brier_score_values = []
    accuracy_values = []
    fig, axes = plt.subplots(figsize=(10, 6), sharey=True)
    
    width = 0.2
    summary = pd.read_csv('../data/processed/rolling_evaluation_summary.csv')
    models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    x = np.arange(len(models) + 1)
    for model in models:
        model_summary = summary[summary['Model'] == model]
        h_values.append(model_summary['Home Wins'].values[0])
        d_values.append(model_summary['Draws'].values[0])
        a_values.append(model_summary['Away Wins'].values[0])

        log_loss_values.append(model_summary['Log Loss'].values[0])
        brier_score_values.append(model_summary['Brier Score'].values[0])
        accuracy_values.append(model_summary['Accuracy'].values[0])
    
    models.append('Actual Results')
    h_values.append(all_seasons_h)
    d_values.append(all_seasons_d)
    a_values.append(all_seasons_a)
    plt.bar(x - width, h_values, width, label='Home Win', color='lightblue')
    plt.bar(x, d_values, width, label='Draw', color='lightgreen')
    plt.bar(x + width, a_values, width, label='Away Win', color='red')
  
    plt.xticks(x, models)
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Match Outcomes by Model')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # plot small multiples of log loss, brier score and accuracy for each model
    models = models[:-1]
    x = np.arange(len(models))
    fig, axes = plt.subplots(figsize=(10, 6), sharey=True)
    plt.bar(x - width, log_loss_values, width, label='Log Loss', color='lightblue')
    plt.bar(x, brier_score_values, width, label='Brier Score', color='lightgreen')
    plt.bar(x + width, accuracy_values, width, label='Accuracy', color='red')
  
    plt.xticks(x, models)
    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.ylim(0, 2)
    plt.title('Metric Comparison of Machine Learning Models (Phase 1)')
    plt.legend()
    plt.tight_layout()
    plt.show()
        
if __name__ == "__main__":
    run()