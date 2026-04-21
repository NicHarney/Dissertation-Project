
def run():
    print("Creating graphs...")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

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

if __name__ == "__main__":
    run()