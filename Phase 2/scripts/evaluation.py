
def run():
    print("Evaluating models...")
    import pandas as pd
    import numpy as np
    import math

    # load all datasets
    dfs = ["matches_with_strengths_predictions", "matches_with_strengths_weighted_predictions", "matches_with_strengths_weighted_opponents_predictions", "matches_with_strengths_weighted_form_predictions"]
    evaluation_table = []

    # evaluate each dataset
    for name in dfs:


        df = pd.read_csv(f'../Data/{name}.csv')
        if name == "matches_with_strengths_predictions":
            model = "Base Model"
        elif name == "matches_with_strengths_weighted_predictions":
            model = "Weighted Model"
        elif name == "matches_with_strengths_weighted_opponents_predictions":
            model = "Opposition factor Model"
        elif name == "matches_with_strengths_weighted_form_predictions":
            model = "Form Model"


        log_loss = []
        brier_score = []
        wins = 0

        matches = 0

        # evaluate each match
        for _, match in df.iterrows():

            # log loss and brier score
            if match['Result'] == "H":
                prob = match['P_home_win']

                brier_row = (np.square(match['P_home_win'] - 1) + np.square(match['P_draw']) + np.square(match['P_away_win']))
                
            elif match['Result'] == "D":
                prob = match['P_draw']
                brier_row = (np.square(match['P_home_win']) + np.square(match['P_draw'] - 1) + np.square(match['P_away_win']))
                
            elif match['Result'] == "A":
                prob = match['P_away_win']
                
                brier_row = (np.square(match['P_home_win']) + np.square(match['P_draw']) + np.square(match['P_away_win'] - 1))

            # general accuracy
            if match['prediction'] == match['Result']:
                wins += 1
            matches += 1
            
            prob = max(prob, 1e-15)  # avoid log(0)
            log_loss_row = -math.log(prob)
            

            brier_score.append(brier_row)
            log_loss.append(log_loss_row)
        total_log_loss = np.mean(log_loss)
        total_brier_score = np.mean(brier_score)



        total_accuracy = (wins / matches ) * 100

        # overall evaluation
        evaluation_row = {}
        evaluation_row.update({
            "Model": model,
            "Season": "all",
            "Log Loss": total_log_loss,
            "Accuracy": total_accuracy,
            "Brier_score": total_brier_score
        })
        evaluation_table.append(evaluation_row)


        seasons = df['season'].unique()
        # evaluate each season separately
        for season in seasons:
            season_df = df[df['season'] == season]
            
            season_predictions = pd.DataFrame()
            season_predictions['Team'] = season_df['home_team'].unique()
            season_predictions['Points'] = 0
            wins = 0
            matches = 0
            log_loss = []
            brier_score = []
            for _, match in season_df.iterrows():

                # log loss and brier score
                if match['Result'] == "H":
                    prob = match['P_home_win']

                    brier_row = (np.square(match['P_home_win'] - 1) + np.square(match['P_draw']) + np.square(match['P_away_win']))
                    
                elif match['Result'] == "D":
                    prob = match['P_draw']
                    brier_row = (np.square(match['P_home_win']) + np.square(match['P_draw'] - 1) + np.square(match['P_away_win']))
                    
                elif match['Result'] == "A":
                    prob = match['P_away_win']
                    
                    brier_row = (np.square(match['P_home_win']) + np.square(match['P_draw']) + np.square(match['P_away_win'] - 1))

                # general accuracy
                if match['prediction'] == match['Result']:
                    wins += 1
                matches += 1
                # allocate points based on predictions
                if match['prediction'] == "H":
                    season_predictions.loc[season_predictions['Team'] == match['home_team'], 'Points'] += 3
                elif match['prediction'] == "D":
                    season_predictions.loc[season_predictions['Team'] == match['home_team'], 'Points'] += 1
                    season_predictions.loc[season_predictions['Team'] == match['away_team'], 'Points'] += 1
                elif match['prediction'] == "A":
                    season_predictions.loc[season_predictions['Team'] == match['away_team'], 'Points'] += 3
                
                log_loss_row = -math.log(prob)
                

                brier_score.append(brier_row)
                log_loss.append(log_loss_row)

            season_predictions = season_predictions.sort_values(by='Points', ascending=False).reset_index(drop=True)
            season_predictions['Rank'] = season_predictions.index + 1
            season_predictions.to_csv(f'../Data/{name}_{season}_predictions.csv', index=False)
            total_log_loss = np.mean(log_loss)
            total_brier_score = np.mean(brier_score)
            total_accuracy = (wins / matches ) * 100

            evaluation_row = {}
            evaluation_row.update({
            "Model": model,
            "Season": season,
            "Log Loss": total_log_loss,
            "Accuracy": total_accuracy,
            "Brier_score": total_brier_score
            })
            evaluation_table.append(evaluation_row)

    new_df = pd.DataFrame(evaluation_table)
    new_df.to_csv('../Data/season_evaluation.csv')
    print("Saved dataset")

if __name__ == "__main__":
    run()