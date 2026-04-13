# Build team profiles by calculating weighted stats


import pandas as pd
import numpy as np
df_all_data = pd.read_csv('../data/processed/optimised_data.csv')


home_stats = df_all_data.groupby(['HomeTeam', 'Div']).agg({
        'FTHG': 'mean', # Home team goals scored
        'FTAG': 'mean', # Home team goals conceded
        'HS': 'mean', # Home team shots
        'AS': 'mean', # Home team shots conceded
        'HST': 'mean', # Home team shots on target
        'AST': 'mean', # Home team shots on target conceded
        'HC': 'mean', # Home team corners
        'AC': 'mean', # Home team corners conceded
    }).rename(columns={
        'FTHG': 'avg_HomeGoalsScored',
        'FTAG': 'avg_HomeGoalsConceded',
        'HS': 'avg_HomeShots',
        'AS': 'avg_HomeShotsConceded',
        'HST': 'avg_HomeShotsOnTarget',
        'AST': 'avg_HomeShotsOnTargetConceded',
        'HC': 'avg_HomeCorners',
        'AC': 'avg_HomeCornersConceded'
    })
home_stats['avg_Home_goal_difference'] = home_stats['avg_HomeGoalsScored'] - home_stats['avg_HomeGoalsConceded']

home_results = df_all_data.groupby(['HomeTeam', 'Div'])['FTR'].agg(
    
    avg_Home_ppg=lambda x: (x == 'H').mean() * 3 + (x == 'D').mean() * 1,
    avg_Home_win_percentage=lambda x: (x == 'H').mean() * 100
   
)

away_stats = df_all_data.groupby(['AwayTeam', 'Div']).agg({
    'FTHG': 'mean', # Away team goals conceded
    'FTAG': 'mean', # Away team goals scored
    'HS': 'mean', # Away team shots conceded
    'AS': 'mean', # Away team shots
    'HST': 'mean', # Away team shots on target conceded
    'AST': 'mean', # Away team shots on target
    'HC': 'mean', # Away team corners conceded
    'AC': 'mean', # Away team corners
}).rename(columns={
    'FTHG': 'avg_AwayGoalsConceded',
    'FTAG': 'avg_AwayGoalsScored',
    'HS': 'avg_AwayShotsConceded',
    'AS': 'avg_AwayShots',
    'HST': 'avg_AwayShotsOnTargetConceded',
    'AST': 'avg_AwayShotsOnTarget',
    'HC': 'avg_AwayCornersConceded',
    'AC': 'avg_AwayCorners',
})

away_stats['avg_Away_goal_difference'] = away_stats['avg_AwayGoalsScored'] - away_stats['avg_AwayGoalsConceded']



away_results = df_all_data.groupby(['AwayTeam', 'Div'])['FTR'].agg(
    avg_Away_ppg=lambda x: (x == 'A').mean() * 3 + (x == 'D').mean() * 1,
    avg_Away_win_percentage=lambda x: (x == 'A').mean() * 100
)


all_stats = pd.concat([home_stats, home_results, away_stats, away_results], axis=1).reset_index().rename(columns={'level_0': 'Team', 'level_1': 'Season'})
all_stats.to_csv('../data/processed/team_profiles.csv', index=False)



