
def run():
    print("Running negative binomial model...")
    import numpy as np
    import pandas as pd
    from scipy.stats import nbinom
    import math


    dfs = ["matches_with_strengths_weighted"]

    for name in dfs:
        df = pd.read_csv(f'../Data/{name}.csv')
        MAX_GOALS = 10

        
        data_rows = []
        # evaluate each match
        for _,match in df.iterrows():

            # calculate league averages prior to this match
            past_df = df[df['date'] < match['date']]
            league_home_avg = past_df['home_xG'].mean() if len(past_df) > 0 else 1
            league_away_avg = past_df['away_xG'].mean() if len(past_df) > 0 else 1

            # calculate xG for each team
            home_xg = (
                match['home_attack'] * 
                match['away_defence'] *
                league_home_avg 
            )

            away_xg = (
                match['away_attack'] *
                match['home_defence'] *
                league_away_avg 
            )
            score_matrix = np.zeros((MAX_GOALS, MAX_GOALS))

            # calculate probability of each scoreline using Negative Binomial distribution
            for h in range(MAX_GOALS):
                for a in range(MAX_GOALS):
                    base_prob = nbinom.pmf(h, match['k'], match['k']/(match['k'] + home_xg)) * nbinom.pmf(a, match['k'], match['k']/(match['k'] + away_xg))
                    score_matrix[h,a] = base_prob

            # normalize
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

            data_row = {}
            data_row.update({
                "date": match['date'],
                "home_team": match['home_team'],
                "away_team": match['away_team'],
                "home_goals": match['home_goals'],
                "away_goals": match['away_goals'],
                "season": match["season"],
                "P_home_win": home_win,
                "P_draw": draw,
                "P_away_win": away_win,
                "prediction": result,
                "Result": actual_result
            })
            data_rows.append(data_row)

        new_df = pd.DataFrame(data_rows)
        new_df.to_csv('../Data/negative_binomial_predictions.csv')
        print("Saved dataset")
if __name__ == "__main__":
    run()