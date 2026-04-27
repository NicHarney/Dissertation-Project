
def run():
    print("Building weighted team profiles...")

    import pandas as pd
    import numpy as np

    df_all_data = pd.read_csv('../data/processed/optimised_data.csv')
    stats = pd.read_csv('../data/processed/team_profiles.csv')



    target_seasons = ['2017-18', '2018-19', '2019-20', '2020-21']

    for season in target_seasons:
        # Filter stats to include only seasons before the target season
        if season == '2017-18':
            all_stats = stats[stats['Div'] == season] 
        else:
            all_stats = stats[stats['Div'] < season] 
        # apply weightings to create a single profile score for each team-season combination
        team_listing = all_stats['Team'].unique()

        team_weightings = pd.DataFrame()
        stat_columns = [col for col in all_stats.columns if col not in ['Team', 'Div', 'ProfileType']]
        results = []


        for team in team_listing:
            # Apply weightings to each stat
            team_data = all_stats[all_stats['Team'] == team].sort_values(by='Div', ascending=True)  

            seasons_count = len(team_data)
            weights = np.arange(1, seasons_count + 1)
            # Normalize weights to sum to 1
            weights = weights / weights.sum()  

            weighted_stats = {}
            weighted_stats['Team'] = team
            # Calculate weighted average for each stat
            for col in stat_columns:
                weighted_stats[col] = (team_data[col].values * weights).sum()

            weighted_stats['most_recent_season'] = team_data['Div'].iloc[-1]
            weighted_stats['seasons_count'] = seasons_count

            results.append(weighted_stats)

        team_weightings = pd.DataFrame(results)
            

        # factor in teams that have just been promoted in the target season and never before
        if season == '2020-21':
            target_data = pd.read_csv('../data/processed/target_data.csv')
            target_teams = target_data['HomeTeam'].unique()
        else:
            target_data = stats[stats['Div'] == season]
            target_teams = target_data['Team'].unique()
        calc_teams = team_weightings['Team'].unique()
    
        # add stats for teams who have not been in the league before
        for team in target_teams:
            if team not in calc_teams:
            
                team_data = all_stats[all_stats['Team'] == team].sort_values(by='Div', ascending=True)

                if len(team_data) == 0:
                    

                    negative_stats = ['avg_HomeGoalsConceded', 'avg_HomeShotsConceded', 'avg_HomeShotsOnTargetConceded', 'avg_HomeCornersConceded',
                                    'avg_AwayGoalsConceded', 'avg_AwayShotsConceded', 'avg_AwayShotsOnTargetConceded', 'avg_AwayCornersConceded']
                    

                    lower_quartiles = team_weightings.select_dtypes(include='number').quantile(0.25)
                    upper_quartiles = team_weightings.select_dtypes(include='number').quantile(0.75)
                    new_row = {}
                    # allocate promoted teams with no premier league history the lower quartile positive stats and upper quartile negative stats
                    for col in stat_columns:
                        if col in negative_stats:
                            new_row[col] = upper_quartiles[col]
                        else:
                            new_row[col] = lower_quartiles[col]
                    
                    new_row['Team'] = team
                    new_row['most_recent_season'] = 'N/A'
                    new_row['seasons_count'] = 0
                    team_weightings = pd.concat([team_weightings, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    # if promoted teams have premier league history, use this data for stats
                    seasons_count = len(team_data)
                    weights = np.arange(1, seasons_count + 1)
                    weights = weights / weights.sum() 

                    weighted_stats = {}
                    weighted_stats['Team'] = team
                    for col in stat_columns:
                        weighted_stats[col] = (team_data[col].values * weights).sum()

                    weighted_stats['most_recent_season'] = team_data['Div'].iloc[-1]
                    weighted_stats['seasons_count'] = seasons_count

                    team_weightings = pd.concat([team_weightings, pd.DataFrame([weighted_stats])], ignore_index=True)

        team_weightings.to_csv(f'../data/processed/team_weighted_profiles_{season}.csv', index=False)


if __name__ == "__main__":
    run()
    



