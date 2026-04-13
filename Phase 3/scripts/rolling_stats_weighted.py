import pandas as pd
import numpy as np
from datetime import timezone
import math


df = pd.read_csv('../Data/epl_all_seasons.csv')
df['date'] = pd.to_datetime(df['date'])


df = df.sort_values('date').reset_index(drop=True)




enriched_rows = []

for i, match in df.iterrows():

    home_team = match['home_team']
    away_team = match['away_team']

    home_xG = match['home_xG']
    away_xG = match['away_xG']

    current_date = match['date']

    past_df = df.iloc[:i]

    past_matches = past_df[past_df['home_team'] == home_team]

    league_home_avg = past_df['home_xG'].mean() if len(past_df) > 0 else 1
    league_away_avg = past_df['away_xG'].mean() if len(past_df) > 0 else 1
    total_weight = 0
    total_xG = 0
    total_xGA = 0



    if len(past_matches) == 0:
        home_attack = 1.0
        home_defence = 1.0
    else:
        for _, past_match in past_matches.iterrows():
            time_diff = (current_date - past_match['date']).days
            weight = math.exp(-0.005 * time_diff)
            total_xG += weight * past_match['home_xG']
            total_xGA += weight * past_match['away_xG']
            total_weight += weight
        
        weighted_avg_xG = total_xG / total_weight
        weighted_avg_xGA = total_xGA / total_weight

        home_attack = weighted_avg_xG / league_home_avg
        home_defence = weighted_avg_xGA / league_away_avg
        
       
    
    away_past_matches = past_df[past_df['away_team'] == away_team]
    total_weight = 0
    total_xG = 0
    total_xGA = 0


    if len(away_past_matches) == 0:
        away_attack = 1.0
        away_defence = 1.0
    else:
        for _, away_past_match in away_past_matches.iterrows():
            time_diff = (current_date - away_past_match['date']).days
            weight = math.exp(-0.005 * time_diff)
            total_xG += weight * away_past_match['away_xG']
            total_xGA += weight * away_past_match['home_xG']
            total_weight += weight
        
        weighted_avg_xG = total_xG / total_weight
        weighted_avg_xGA = total_xGA / total_weight

        away_attack = weighted_avg_xG / league_away_avg
        away_defence = weighted_avg_xGA / league_home_avg
        
    goals = pd.concat([past_df['home_goals'], past_df['away_goals']])
    mean_goals = goals.mean()
    variance_goals = goals.var()
    #negative binomial k parameter estimation
    k =  (mean_goals**2) / (variance_goals - mean_goals) if variance_goals > mean_goals else 1000
    #laplace b parameter estimation
    b = math.sqrt(variance_goals / 2) if variance_goals > 0 else 1.0


    enriched_row = match.to_dict()
    enriched_row.update({
        "home_attack": home_attack,
        "home_defence": home_defence,
        "away_attack": away_attack,
        "away_defence": away_defence,
        "mean_goals": mean_goals,
        "variance_goals": variance_goals,
        "k": k,
        "b": b
    })

    enriched_rows.append(enriched_row)
# save match strengths to csv
new_df = pd.DataFrame(enriched_rows)
new_df.to_csv("../Data/matches_with_strengths_weighted.csv")

print("Saved dataset")
 
    

    