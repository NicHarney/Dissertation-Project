
def run():
    print("Running Bayesian Poisson model...")
 
    import pandas as pd
    import numpy as np
    from scipy.stats import poisson
    import math
    from datetime import timezone

    MAX_GOALS = 10

    df = pd.read_csv('../Data/epl_all_seasons.csv')
    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values('date').reset_index(drop=True)

    team_stats = {}
    enriched_rows = []
    # implement bayesian poisson model
    for i, match in df.iterrows():

        home_team = match['home_team']
        away_team = match['away_team']
    
        # find prior matches
        current_date = match['date']
        past_df = df.iloc[:i]

        league_home_avg = past_df['home_xG'].mean() if len(past_df) > 0 else 1
        league_away_avg = past_df['away_xG'].mean() if len(past_df) > 0 else 1

        learning_rate = 0.05
    
        # initialize team stats if not present
        if home_team not in team_stats:
            team_stats[home_team] = {
                "home_matches": 0,
                "away_matches": 0,
                "attack_mean_home": 1.0,
                "attack_mean_away": 1.0,
                "defence_mean_home": 1.0,
                "defence_mean_away": 1.0,
                "home_xG_sum": 0.0,
                "away_xG_sum": 0.0,
                "home_xGA_sum": 0.0,
                "away_xGA_sum": 0.0
            }
        if away_team not in team_stats:
            team_stats[away_team] = {
            "home_matches": 0,
            "away_matches": 0,
            "attack_mean_home": 1.0,
            "attack_mean_away": 1.0,
            "defence_mean_home": 1.0,
            "defence_mean_away": 1.0,
            "home_xG_sum": 0.0,
            "away_xG_sum": 0.0,
            "home_xGA_sum": 0.0,
            "away_xGA_sum": 0.0
            
        }

        home_stats = team_stats[home_team]
        away_stats = team_stats[away_team]

        # calculate attack and defence strenghts
        if home_stats['home_matches'] > 0:
            home_avg_for = home_stats['home_xG_sum'] / home_stats['home_matches']
            home_avg_against = home_stats['home_xGA_sum'] / home_stats['home_matches']
        else:
            home_avg_for = league_home_avg
            home_avg_against = league_away_avg
        if away_stats['away_matches'] > 0:
            away_avg_for = away_stats['away_xG_sum'] / away_stats['away_matches']
            away_avg_against = away_stats['away_xGA_sum'] / away_stats['away_matches']
        else:
            away_avg_for = league_away_avg
            away_avg_against = league_home_avg

        # calculate expected goals
        home_xg = (
            home_avg_for * 
            away_avg_against /
            league_home_avg *
            home_stats['attack_mean_home'] * 
            away_stats['defence_mean_away'] 
        
        )
        away_xg = (
            away_avg_for *
            home_avg_against / 
            league_away_avg *
            away_stats['attack_mean_away'] *
            home_stats['defence_mean_home'] 
            
        )
    
        # initialize score matrix
        score_matrix = np.zeros((MAX_GOALS, MAX_GOALS))

        # calculate probability of each scoreline using Poisson distribution
        for h in range(MAX_GOALS):
            for a in range(MAX_GOALS):
                prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                score_matrix[h,a] = prob

        # normalize score matrix
        score_matrix /= score_matrix.sum()

        home_win = 0
        draw = 0
        away_win = 0

        # find most likely result
        for h in range(MAX_GOALS):
            for a in range(MAX_GOALS):

                prob = score_matrix[h,a]

                if h > a:
                    home_win += prob
                elif h == a:
                    draw += prob
                else:
                    away_win += prob
                
        # prediction 
        if home_win >= away_win:
            if home_win >= draw:
                result = "H"
            else:
                result = "D"
        elif away_win > home_win:
            if away_win >= draw:
                result = "A"
            else:
                result = "D"

        # actual result
        if match['home_goals'] > match['away_goals']:
            actual_result = "H"
        elif match['home_goals'] < match['away_goals']:
            actual_result = "A"
        elif match['home_goals'] == match['away_goals']:
            actual_result = "D"


        # update enriched row
        enriched_row = {}
        enriched_row.update({
            "date": match['date'],
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": match['home_goals'],
            "away_goals": match['away_goals'],
            "season": match['season'],
            "attack_mean_home": home_stats['attack_mean_home'],
            "attack_mean_away": away_stats['attack_mean_away'],
            "defence_mean_home": home_stats['defence_mean_home'],
            "defence_mean_away": away_stats['defence_mean_away'],
            "P_home_win": home_win,
            "P_draw": draw,
            "P_away_win": away_win,
            "prediction": result,
            "Result": actual_result
        })
        enriched_rows.append(enriched_row)

        # update team stats
        home_stats['home_matches'] += 1
        away_stats['away_matches'] += 1

        # update home team attack and defence
        error_home = match['home_xG'] - home_xg
        home_stats['attack_mean_home'] += learning_rate * error_home
        home_stats['defence_mean_home'] += learning_rate * (match['away_xG'] - away_xg)

        # clamp home attack and defence means to be non-negative
        home_stats['attack_mean_home'] = max(0.01, home_stats['attack_mean_home'])
        home_stats['defence_mean_home'] = max(0.01, home_stats['defence_mean_home'])

        # update away team attack and defence
        error_away = match['away_xG'] - away_xg
        away_stats['attack_mean_away'] += learning_rate * error_away
        away_stats['defence_mean_away'] += learning_rate * (match['home_xG'] - home_xg)

        # clamp away attack and defence means to be non-negative
        away_stats['attack_mean_away'] = max(0.01, away_stats['attack_mean_away'])
        away_stats['defence_mean_away'] = max(0.01, away_stats['defence_mean_away'])

        # update xG sums
        home_stats['home_xG_sum'] += match['home_xG']
        away_stats['away_xG_sum'] += match['away_xG']
        home_stats['home_xGA_sum'] += match['away_xG']
        away_stats['away_xGA_sum'] += match['home_xG']

    new_df = pd.DataFrame(enriched_rows)
    new_df.to_csv('../Data/bayesian_predictions.csv', index=False)
    print("Predictions saved to bayesian_predictions.csv")
if __name__ == "__main__":
    run()