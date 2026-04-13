import pandas as pd
import numpy as np


target_seasons = ['2017-18', '2018-19', '2019-20', '2020-21']

for season in target_seasons:
    if season == '2020-21':
        target_dataset = pd.read_csv('../data/processed/target_data.csv')

    else:
        target_dataset = pd.read_csv('../data/processed/optimised_data.csv')
        target_dataset = target_dataset[target_dataset['Div'] == season]
    profiles = pd.read_csv(f'../data/processed/team_weighted_profiles_{season}.csv')


    features = pd.DataFrame()


    # Prepare the profiles data
    home_cols = profiles.filter(regex='^avg_Home').columns
    away_cols = profiles.filter(regex='^avg_Away').columns
    
    # split dataframes 
    home_df = profiles[['Team'] + home_cols.tolist()].copy()
    away_df = profiles[['Team'] + away_cols.tolist()].copy()

    # Merge with target dataset
    features = target_dataset.merge(
        home_df, 
        left_on='HomeTeam', 
        right_on='Team', 
        how='left'
    ).drop('Team', axis=1)

    features = features.merge(
        away_df,
        left_on='AwayTeam',
        right_on='Team',
        how='left'
    ).drop('Team', axis=1)

    # Calculate differentials
    for home_col in home_cols:
        
        #Extract necessary columns
        stat_name = home_col.replace('avg_Home', '')
        
        
        away_col = f'avg_Away{stat_name}'
        
        # Only calculate differential if corresponding away stat exists
        if away_col in away_cols:
            
            features[f'{stat_name}Differential'] = features[home_col] - features[away_col]

    # Identify the differential columns
    differential_cols = [f'{col.replace("avg_Home", "")}Differential' for col in home_cols 
                        if f'avg_Away{col.replace("avg_Home", "")}' in away_cols]

    # add necessary additional columns
    features = features[['Date', 'HomeTeam', 'AwayTeam', 'Div'] + differential_cols]

    # add target win/draw/loss column
    features['FTR'] = target_dataset['FTR'].values
    
    features.to_csv(f'../data/processed/features_{season}.csv', index=False)
