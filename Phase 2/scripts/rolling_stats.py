import pandas as pd
import numpy as np


df_all_data = pd.read_csv('../Data/epl_all_seasons.csv')
df_all_data['date'] = pd.to_datetime(df_all_data['date'])


df_all_data = df_all_data.sort_values('date').reset_index(drop=True)



team_stats = {}
enriched_rows = []

for _, match in df_all_data.iterrows():


    # league averages prior to this match
    past_df = df_all_data[df_all_data['date'] < match['date']]
    league_home_avg = past_df['home_xG'].mean() if len(past_df) > 0 else 1
    league_away_avg = past_df['away_xG'].mean() if len(past_df) > 0 else 1

    
    home_team = match['home_team']
    away_team = match['away_team']

    home_xG = match['home_xG']
    away_xG = match['away_xG']


    if home_team not in team_stats:
        team_stats[home_team] = {
            "home_matches": 0,
            "away_matches": 0,
            "home_xG_sum": 0.0,
            "away_xG_sum": 0.0,
            "home_xGA_sum": 0.0,
            "away_xGA_sum": 0.0,
        }
    if away_team not in team_stats:
           team_stats[away_team] = {
            "home_matches": 0,
            "away_matches": 0,
            "home_xG_sum": 0.0,
            "away_xG_sum": 0.0,
            "home_xGA_sum": 0.0,
            "away_xGA_sum": 0.0,
        }

    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]

    # calculate strengths

    if home_stats['home_matches'] == 0:
        home_attack = 1.0
        home_defence = 1.0
    else:
        home_attack = (home_stats['home_xG_sum'] / home_stats['home_matches']) / league_home_avg
        home_defence = (home_stats['home_xGA_sum'] / home_stats['home_matches']) / league_away_avg
    
    if away_stats['away_matches'] == 0:
        away_attack = 1.0
        away_defence = 1.0
    else:
        away_attack = (away_stats['away_xG_sum'] / away_stats['away_matches']) / league_away_avg
        away_defence = (away_stats['away_xGA_sum'] / away_stats['away_matches']) / league_home_avg

    # store enriched row
    enriched_row = match.to_dict()
    enriched_row.update({
        "home_attack": home_attack,
        "home_defence": home_defence,
        "away_attack": away_attack,
        "away_defence": away_defence
    })

    enriched_rows.append(enriched_row)

    # update team stats
    home_stats["home_matches"] += 1
    home_stats["home_xG_sum"] += home_xG
    home_stats["home_xGA_sum"] += away_xG

    away_stats["away_matches"] += 1
    away_stats["away_xG_sum"] += away_xG
    away_stats["away_xGA_sum"] += home_xG


# save match strengths to csv
new_df = pd.DataFrame(enriched_rows)
new_df.to_csv("../Data/matches_with_strengths.csv")

print("Saved dataset")
 
    

    