import pandas as pd
import numpy as np
import math


dfs = ["negative_binomial_predictions", "laplace_predictions", "bayesian_predictions"]
evaluation_table = []

for name in dfs:


    df = pd.read_csv(f'../Data/{name}.csv')
    if name == "negative_binomial_predictions":
        model = "Negative Binomial"
    elif name == "laplace_predictions":
        model = "Laplace"
    elif name == "bayesian_predictions":
        model = "Bayesian"

    log_loss = []
    brier_score = []
    wins = 0

    matches = 0

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

    for season in seasons:
        season_df = df[df['season'] == season]
        
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
            
            
            log_loss_row = -math.log(prob)
            

            brier_score.append(brier_row)
            log_loss.append(log_loss_row)

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
