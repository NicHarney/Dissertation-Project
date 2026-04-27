
def run():
    print("Loading dataset...")
    from understatapi import UnderstatClient
    import pandas as pd

    seasons = ["2017", "2018", "2019", "2020"]
    all_data = []

    # Use understat API to access required data
    with UnderstatClient() as client:
        league = client.league(league="EPL")

        # separate data into seasons
        for season in seasons:
            print(f"Downloading {season}...")

            results = league.get_match_data(season=season)

            rows = []
            # Extract necessary data for models
            for match in results:
                rows.append({
                    "date": match["datetime"],
                    "home_team": match["h"]["title"],
                    "away_team": match["a"]["title"],
                    "home_goals": int(match["goals"]["h"]),
                    "away_goals": int(match["goals"]["a"]),
                    "home_xG": float(match["xG"]["h"]),
                    "away_xG": float(match["xG"]["a"]),
                    "season": season
                })
            
            df = pd.DataFrame(rows)
            
            df.to_csv(f"../Data/epl_{season}.csv", index=False)
            all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv("../Data/epl_all_seasons.csv", index=False)

    print("Saved all seasons.")
if __name__ == "__main__":
    run()