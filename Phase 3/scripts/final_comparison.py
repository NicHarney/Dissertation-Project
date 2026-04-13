import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_phase1 = pd.read_csv('../../Phase 1/data/processed/rolling_evaluation_results.csv')
df_phase2 = pd.read_csv('../../Phase 2/Data/season_evaluation.csv')
df_phase3 = pd.read_csv('../Data/season_evaluation.csv')

# fix season formatting in phase 1
df_phase1 = df_phase1[df_phase1['Model'] == "Logistic Regression"]  # only keep one model for phase 1
df_phase1 = df_phase1.groupby('Model').agg({
    'Log Loss': ['mean', 'std'],
    'Brier Score': ['mean', 'std'],
    'Accuracy': 'mean',
}).reset_index()
df_phase1.rename(columns={'Brier Score': 'Brier_score'}, inplace=True)
df_phase1.columns = ['Model', 'Log Loss', 'Log Loss Std', 'Brier Score', 'Brier Score Std', 'Accuracy']


df_phase2 = df_phase2[df_phase2['Model'] == "Weighted Model"]  
df_phase2 = df_phase2.groupby('Model').agg({
    'Log Loss': ['mean', 'std'],
    'Brier_score': ['mean', 'std'],
    'Accuracy': 'mean',
}).reset_index()
df_phase2.columns = ['Model', 'Log Loss', 'Log Loss Std', 'Brier Score', 'Brier Score Std', 'Accuracy']


df_phase3 = df_phase3.groupby('Model').agg({
    'Log Loss': ['mean', 'std'],
    'Brier_score': ['mean', 'std'],
    'Accuracy': 'mean',
}).reset_index()
df_phase3.columns = ['Model', 'Log Loss', 'Log Loss Std', 'Brier Score', 'Brier Score Std', 'Accuracy']


df = pd.concat([df_phase1, df_phase2, df_phase3], ignore_index=True)
df.to_csv('../Data/final_comparison.csv', index=False)


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