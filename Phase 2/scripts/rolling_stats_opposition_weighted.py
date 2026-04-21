
def run():
    print("Calculating rolling stats with opponent weighting...")
    
    import pandas as pd
    import numpy as np
    import math

    # Load weighted strengths 
    df = pd.read_csv('../Data/matches_with_strengths_weighted.csv')
    df['date'] = pd.to_datetime(df['date'])

    
    df = df.sort_values('date').reset_index(drop=True)


    enriched_rows = []

    # calculate each match's strengths with opponent weighting
    for i, match in df.iterrows():

        home_team = match['home_team']
        away_team = match['away_team']
        current_date = match['date']

        # Only use past matches 
        past_df = df.iloc[:i]

        past_matches = past_df[past_df['home_team'] == home_team]

        league_home_avg = past_df['home_xG'].mean() if len(past_df) > 0 else 1
        league_away_avg = past_df['away_xG'].mean() if len(past_df) > 0 else 1

        total_weight = 0
        total_xG = 0
        total_xGA = 0

        # Calculate attack and defence strengths with opponent weighting
        if len(past_matches) == 0:
            home_attack = 1.0
            home_defence = 1.0
        else:
            for _, past_match in past_matches.iterrows():

                time_diff = (current_date - past_match['date']).days
                weight = math.exp(-0.005 * time_diff)

                
                opponent_defence = past_match.get('away_defence', 1.0)

                # clamp + shrink to prevent extreme adjustments
                opponent_defence = max(0.7, min(opponent_defence, 1.3))
                opponent_defence = 0.5 * opponent_defence + 0.5

                
                adjusted_xG = past_match['home_xG'] / opponent_defence

                
                adjusted_xGA = past_match['away_xG']

                total_xG += weight * adjusted_xG
                total_xGA += weight * adjusted_xGA
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
            for _, past_match in away_past_matches.iterrows():

                time_diff = (current_date - past_match['date']).days
                weight = math.exp(-0.005 * time_diff)

                
                opponent_defence = past_match.get('home_defence', 1.0)

                # clamp + shrink to prevent extreme adjustments
                opponent_defence = max(0.7, min(opponent_defence, 1.3))
                opponent_defence = 0.5 * opponent_defence + 0.5

                
                adjusted_xG = past_match['away_xG'] / opponent_defence

                
                adjusted_xGA = past_match['home_xG']

                total_xG += weight * adjusted_xG
                total_xGA += weight * adjusted_xGA
                total_weight += weight

            weighted_avg_xG = total_xG / total_weight
            weighted_avg_xGA = total_xGA / total_weight

            away_attack = weighted_avg_xG / league_away_avg
            away_defence = weighted_avg_xGA / league_home_avg
            
    
        enriched_row = match.to_dict()
        enriched_row.update({
            "home_attack": home_attack,
            "home_defence": home_defence,
            "away_attack": away_attack,
            "away_defence": away_defence
        })

        enriched_rows.append(enriched_row)

    
    new_df = pd.DataFrame(enriched_rows)
    new_df.to_csv("../Data/matches_with_strengths_weighted_opponents.csv", index=False)

    print("Saved dataset")
if __name__ == "__main__":
    run()