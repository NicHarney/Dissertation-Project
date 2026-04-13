# Load in data
# Retrieve Phase 1 important features
# Create new dataset with only important features

import pandas as pd

# Load in data
df_2017_18 = pd.read_excel('../data/raw/2017-18.xlsx')
df_2017_18['Div'] = '2017-18' # Add season column to identify the season for each row
df_2018_19 = pd.read_excel('../data/raw/2018-2019.xlsx')
df_2018_19['Div'] = '2018-19' # Add season column to identify the season for each row
df_2019_20 = pd.read_excel('../data/raw/2019-20.xlsx')
df_2019_20['Div'] = '2019-20' # Add season column to identify the season for each row
df_2020_21 = pd.read_excel('../data/raw/2020-21.xlsx')
df_2020_21['Div'] = '2020-21' # Add season column to identify the season for each row

df_champ_2019_20 = pd.read_excel('../data/raw/Champ-2019-20.xlsx')
df_champ_2019_20 = df_champ_2019_20.loc[:, :'AR'] # Drop betting odds columns between 'B365' and 'PSCA' if they exist
df_champ_2018_19 = pd.read_excel('../data/raw/Champ-2018-19.xlsx')
df_champ_2018_19 = df_champ_2018_19.loc[:, :'AR'] # Drop betting odds columns between 'B365' and 'PSCA' if they exist
df_champ_2017_18 = pd.read_excel('../data/raw/Champ-2017-18.xlsx')
df_champ_2017_18 = df_champ_2017_18.loc[:, :'AR'] # Drop betting odds columns between 'B365' and 'PSCA' if they exist
df_champ = pd.concat([df_champ_2017_18, df_champ_2018_19, df_champ_2019_20])
df_champ.to_csv('../data/processed/champ_data.csv', index=False)

df_target_data = df_2020_21.loc[:, :'AR']
df_target_data.to_csv('../data/processed/target_data.csv', index=False)
df_processed_data = pd.DataFrame()

# Retrieve Phase 1 important features
list_of_dfs = [df_2017_18, df_2018_19, df_2019_20]

# concatenate all dataframes into one processed dataframe
for df in list_of_dfs:
    df_processed_data = pd.concat([df_processed_data, df])

# save the processed data to a new CSV file
df_processed_data.to_csv('../data/processed/processed_data.csv', index=False)

# Remove betting odds columns to create optimised dataset for modelling

# Drop betting odds columns between 'B365' and 'PSCA' if they exist
df_optimised_data = df_processed_data.loc[:, :'AR']


df_optimised_data.to_csv('../data/processed/optimised_data.csv', index=False)
