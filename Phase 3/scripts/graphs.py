
def run():
    print("Generating graphs for evaluation results...")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

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

if __name__ == "__main__":
    run()