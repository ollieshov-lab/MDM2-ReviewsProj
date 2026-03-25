import kagglehub
import pandas as pd

# Download reviews dataset
path = kagglehub.dataset_download("datafiniti/hotel-reviews")
print("Path to dataset files:", path)

# Load the CSV (kagglehub returns a directory path)
import os
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

print(f"Loaded {len(df):,} rows")
print(df.columns.tolist())

# Parse review date and assign seasons
df['Review_Date'] = pd.to_datetime(df['reviews.date'], errors='coerce')
df['Month'] = df['Review_Date'].dt.month

def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter (Dec-Feb)'
    elif month in [3, 4, 5]:
        return 'Spring (Mar-May)'
    elif month in [6, 7, 8]:
        return 'Summer (Jun-Aug)'
    else:
        return 'Autumn (Sep-Nov)'

df['Season'] = df['Month'].apply(assign_season)

# Split into separate DataFrames per season
season_dfs = {season: group.reset_index(drop=True) for season, group in df.groupby('Season')}

winter_df = season_dfs.get('Winter (Dec-Feb)')
spring_df = season_dfs.get('Spring (Mar-May)')
summer_df = season_dfs.get('Summer (Jun-Aug)')
autumn_df = season_dfs.get('Autumn (Sep-Nov)')

# Summary
print(df['Season'].value_counts().sort_index())