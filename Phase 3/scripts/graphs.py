
def run():
    print("Generating graphs for evaluation results...")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from collections import Counter
    df = pd.read_csv('../Data/season_evaluation.csv')

    base_model = pd.read_csv('../../Phase 2/Data/season_evaluation.csv')

    base_model = base_model[base_model['Model'] == 'Base Model']

    # Append base model to df for groupby
    df = pd.concat([df, base_model], ignore_index=True)

    # Line graph across seasons
    pivot = df.pivot(index='Season', columns='Model', values='Log Loss')

    plt.figure()

    for model in pivot.columns:
        plt.plot(pivot.index, pivot[model], marker='o', label=model)

    plt.xlabel('Season')
    plt.ylabel('Log Loss')
    plt.title("Log Loss Across Seasons for each Model")
    plt.legend()
    plt.show()

    # Log loss and brier score averages bar graph

    df = df.groupby('Model').agg({
        'Log Loss': ['mean', 'std'],
        'Brier_score': ['mean', 'std']
    }).reset_index()

    df.columns = ['Model', 'Log Loss', 'Log Loss Std', 'Brier Score', 'Brier Score Std']

    df = df.sort_values('Log Loss')

    # Plotting Log Loss
    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df['Log Loss'], yerr=df['Log Loss Std'],capsize=5, color='skyblue')

    plt.title(f'Log Loss comparison of distribution assumption models (Phase 3)')
    plt.xlabel('Model')
    plt.ylabel('Log Loss')
    plt.ylim(min((df['Log Loss']) - df['Log Loss Std']) * 0.9, max((df['Log Loss']) + df['Log Loss Std']) * 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Get base model values for improvement calculation
    base_log_loss = df[df['Model'] == 'Base Model']['Log Loss'].values[0]
    base_brier_score = df[df['Model'] == 'Base Model']['Brier Score'].values[0]

    # Plotting Brier Score
    df = df.sort_values('Brier Score')
    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df['Brier Score'], yerr=df['Brier Score Std'],capsize=5, color='salmon')
    plt.title(f'Brier Score comparison of distribution assumption models (Phase 3)')
    plt.xlabel('Model')
    plt.ylabel('Brier Score')
    plt.ylim(min((df['Brier Score']) - df['Brier Score Std']) * 0.9, max((df['Brier Score']) + df['Brier Score Std']) * 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # plot improvement versus base model (Poisson) for log loss and brier score
    df['Log Loss Improvement'] = -(df['Log Loss'] - base_log_loss)
    df['Brier Score Improvement'] = -(df['Brier Score'] - base_brier_score)
    df = df[df['Model'] != 'Base Model']
    # Plotting Log Loss Improvement
    df = df.sort_values('Log Loss Improvement')
    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df['Log Loss Improvement'], color='hotpink')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2, label='base model')
    plt.title(f'Log Loss Improvement over Base Model (Phase 3)')
    plt.xlabel('Model')
    plt.ylabel('Log Loss Improvement')
    plt.ylim(min(df['Log Loss Improvement']) * 1.1, max(df['Log Loss Improvement']) * 1.1)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plotting Brier Score Improvement
    df = df.sort_values('Brier Score Improvement')
    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df['Brier Score Improvement'], color='lightgreen')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=2, label='base model')
    plt.title(f'Brier Score Improvement over Base Model (Phase 3)')
    plt.xlabel('Model')
    plt.ylabel('Brier Score Improvement')
    plt.ylim(min(df['Brier Score Improvement']) * 1.1, max(df['Brier Score Improvement']) * 1.1)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    
    # plot small multiples of home win, draw and away win for predicted results each model
    
    csvs = ['laplace_predictions', 'negative_binomial_predictions', 'bayesian_predictions']
    
    all_seasons = pd.read_csv('../Data/laplace_predictions.csv')['Result']
    width = 0.2
    models = ['Laplace', 'Negative Binomial', 'Bayesian Poisson']
    h_values = []
    d_values = []
    a_values = []
    fig = plt.subplots(figsize=(10, 6), sharey=True)
    x = np.arange(len(csvs) + 1)
    for csv in csvs:
        
        model = pd.read_csv(f'../Data/{csv}.csv')['prediction']
    
        counts = Counter(model)
        
        h_values.append(counts['H'])
        d_values.append(counts['D'])
        a_values.append(counts['A'])
        
    counts = Counter(all_seasons)
    h_values.append(counts['H'])
    d_values.append(counts['D'])
    a_values.append(counts['A'])
        
    models.append('Actual Results')
    plt.bar(x - width, h_values, width, label='Home Wins', color='lightblue')
    plt.bar(x, d_values, width, label='Draws', color='lightgreen')
    plt.bar(x + width, a_values, width, label='Away Wins', color='red')

    plt.xticks(x, models, rotation=45)
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.title('Predicted Results by Model')
    plt.legend()
    plt.tight_layout()
    plt.show()


     # plot log loss, brier score and accuracy simple multiples graph
    width = 0.2
    df = pd.read_csv(f'../Data/season_evaluation.csv')
    seasons = df['Season'].unique()

    base_model = pd.read_csv('../../Phase 2/Data/season_evaluation.csv')

    base_df = base_model[base_model['Model'] == 'Base Model']
    models = ['Laplace', 'Negative Binomial', 'Bayesian']
    
    fig, axes = plt.subplots(1,5,figsize=(10, 6), sharey=True)
    x = np.arange(len(models))
    for ax,season in zip(axes,seasons):
        
        season_df = df[df['Season'] == season]
        base_dfseason = base_df[base_df['Season'] == season]
        base_log_loss = base_dfseason['Log Loss'].values[0]
        base_brier_score = base_dfseason['Brier_score'].values[0]
        base_accuracy = base_dfseason['Accuracy'].values[0]/100
        log_loss_values = []
        brier_score_values = []
        accuracy_values = []
        
        for model in models:
        
            modeldf = season_df[season_df['Model'] == model]
        
            
            log_loss_values.append(base_log_loss- modeldf['Log Loss'].values[0])
            brier_score_values.append(base_brier_score - modeldf['Brier_score'].values[0])
            accuracy_values.append(modeldf['Accuracy'].values[0]/100 - base_accuracy)
        
        
        ax.bar(x - width, log_loss_values, width, label='Log Loss', color='lightblue')
        ax.bar(x, brier_score_values, width, label='Brier Score', color='lightgreen')
        ax.bar(x + width, accuracy_values, width, label='Accuracy', color='red')

        ax.set_xticks(x, models, rotation=45)
        ax.set_xlabel('Model')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, label='base model')
        ax.set_ylabel('Metric score')
        ax.set_ylim(-0.05,0.05)
        ax.set_title(f'{season} Season Improvement')
        ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()