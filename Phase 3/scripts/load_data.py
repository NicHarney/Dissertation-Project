
def run():
    print("Loading dataset...")
    import asyncio
    import aiohttp
    import nest_asyncio
    import pandas as pd
    from understat import Understat

    # allow nested event loops 
    nest_asyncio.apply()

    # function to get data for a single season
    async def get_season_data(season):

        # create aiohttp session and understat client
        async with aiohttp.ClientSession() as session:
            understat = Understat(session)
            
            # get match results for the season
            results = await understat.get_league_results("epl", season)

            data = []
            # process each match and extract relevant data
            for match in results:
                data.append({
                    "date": match["datetime"],
                    "home_team": match["h"]["title"],
                    "away_team": match["a"]["title"],
                    "home_goals": int(match["goals"]["h"]),
                    "away_goals": int(match["goals"]["a"]),
                    "home_xG": float(match["xG"]["h"]),
                    "away_xG": float(match["xG"]["a"]),
                    "season": season
                })

            return pd.DataFrame(data)


    # function to download multiple seasons and save to CSV
    def download_seasons(seasons):
        all_data = []

        # create event loop for async calls
        loop = asyncio.get_event_loop()
        
        # download each season and save individual CSVs
        for season in seasons:
            print(f"Downloading season {season}...")
            df = loop.run_until_complete(get_season_data(season))
            df.to_csv(f"../Data/epl_{season}.csv", index=False)
            all_data.append(df)

        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv("../Data/epl_all_seasons.csv", index=False)

        print("Done. Saved individual and combined CSVs.")


    if __name__ == "__main__":
        seasons = ["2017","2018", "2019", "2020"]
        download_seasons(seasons)
if __name__ == "__main__":
    run()