
def run():
    print("Calculating rolling form stats...")
    import pandas as pd
    import numpy as np
    from datetime import timezone
    import math


    df = pd.read_csv('../Data/epl_all_seasons.csv')
    df['date'] = pd.to_datetime(df['date'])


    df = df.sort_values('date').reset_index(drop=True)




    enriched_rows = []

    # calculate rolling stats with weighted averages and form adjustments
    for i, match in df.iterrows():

        home_team = match['home_team']
        away_team = match['away_team']

        home_xG = match['home_xG']
        away_xG = match['away_xG']

        current_date = match['date']

        # calculate league averages prior to this match
        past_df = df.iloc[:i]

        past_matches = past_df[past_df['home_team'] == home_team]

        league_home_avg = past_df['home_xG'].mean() if len(past_df) > 0 else 1
        league_away_avg = past_df['away_xG'].mean() if len(past_df) > 0 else 1
        total_weight = 0
        total_xG = 0
        total_xGA = 0


        # calculate attack and defence strenghts
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
            


        # calculate form adjustments based on last 10 matches
        last_10_home = past_df[(past_df['home_team'] == home_team) | (past_df['away_team'] == home_team)]

        last_10_away = past_df[(past_df['home_team'] == away_team) | (past_df['away_team'] == away_team)]

        if len(last_10_home) > 9:
            
            form_xG = 0
            form_xGA = 0
            form_matches = 0
            for _, form_match in last_10_home.iterrows():
                if form_match['home_team'] == home_team:
                    form_xG += form_match['home_xG']
                    form_xGA += form_match['away_xG']
                else:
                    form_xG += form_match['away_xG']
                    form_xGA += form_match['home_xG']
                form_matches += 1
            form_home_attack = (form_xG / form_matches) / league_home_avg
            form_home_defence = (form_xGA / form_matches) / league_away_avg

            home_attack = (1 - 0.3) * home_attack + (0.3 * form_home_attack)
            home_defence = (1 - 0.3) * home_defence + (0.3 * form_home_defence)


        if len(last_10_away) > 9:
            
            form_xG = 0
            form_xGA = 0
            form_matches = 0
            for _, form_match in last_10_away.iterrows():
                if form_match['home_team'] == home_team:
                    form_xG += form_match['home_xG']
                    form_xGA += form_match['away_xG']
                else:
                    form_xG += form_match['away_xG']
                    form_xGA += form_match['home_xG']
                form_matches += 1
            form_away_attack = (form_xG / form_matches) / league_away_avg
            form_away_defence = (form_xGA / form_matches) / league_home_avg

            away_attack = (1 - 0.3) * away_attack + (0.3 * form_away_attack)
            away_defence = (1 - 0.3) * away_defence + (0.3 * form_away_defence)


        enriched_row = match.to_dict()
        enriched_row.update({
            "home_attack": home_attack,
            "home_defence": home_defence,
            "away_attack": away_attack,
            "away_defence": away_defence
        })

        enriched_rows.append(enriched_row)
    # save match strengths to csv
    new_df = pd.DataFrame(enriched_rows)
    new_df.to_csv("../Data/matches_with_strengths_weighted_form.csv")

    print("Saved dataset")
if __name__ == "__main__":
    run()
 
    

    