import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('../Data/season_evaluation.csv')

# Line plot for Log Loss across seasons for Poisson model

pivot = df.pivot(index='Season', columns='Model', values='Log Loss')

plt.figure()

for model in pivot.columns:
    plt.plot(pivot.index, pivot[model], marker='o', label=model)

plt.xlabel('Season')
plt.ylabel('Log Loss')
plt.title("Log Loss Across Seasons for Poisson Model")
plt.legend()
plt.show()

# Bar graph for average log loss of each model
df = df[pd.notna(df['Log Loss'])]  # Remove rows with NaN log loss

df = df.groupby('Model').agg({
    'Log Loss': ['mean', 'std'],
    'Brier_score': ['mean', 'std']
}).reset_index()

df.columns = ['Model', 'Log Loss', 'Log Loss Std', 'Brier Score', 'Brier Score Std']

df = df.sort_values('Log Loss')

# Plotting Log Loss
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['Log Loss'], yerr=df['Log Loss Std'],capsize=5, color='skyblue')

plt.title(f'Log Loss comparison of Poisson Models (Phase 2)')
plt.xlabel('Model')
plt.ylabel('Log Loss')
plt.ylim(min((df['Log Loss']) - df['Log Loss Std']) * 0.9, max((df['Log Loss']) + df['Log Loss Std']) * 1.1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting Brier Score
df = df.sort_values('Brier Score')
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['Brier Score'], yerr=df['Brier Score Std'],capsize=5, color='salmon')
plt.title(f'Brier Score comparison of Poisson Models (Phase 2)')
plt.xlabel('Model')
plt.ylabel('Brier Score')
plt.ylim(min((df['Brier Score']) - df['Brier Score Std']) * 0.9, max((df['Brier Score']) + df['Brier Score Std']) * 1.1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot improvement versus base model (Poisson) for log loss and brier score
base_model = 'Base Model'

df['Log Loss Improvement'] = -(df['Log Loss'] - df[df['Model'] == base_model]['Log Loss'].values[0])
df['Brier Score Improvement'] = -(df['Brier Score'] - df[df['Model'] == base_model]['Brier Score'].values[0])
df = df[df['Model'] != base_model]
# Plotting Log Loss Improvement
df = df.sort_values('Log Loss Improvement')
plt.figure(figsize=(10, 6))
plt.bar(df['Model'], df['Log Loss Improvement'], color='hotpink')
plt.axhline(y=0, color='black', linestyle='-', linewidth=2, label='base model')
plt.title(f'Log Loss Improvement over Base Model (Phase 2)')
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
plt.title(f'Brier Score Improvement over Base Model (Phase 2)')
plt.xlabel('Model')
plt.ylabel('Brier Score Improvement')
plt.ylim(min(df['Brier Score Improvement']) * 1.1, max(df['Brier Score Improvement']) * 1.1)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()