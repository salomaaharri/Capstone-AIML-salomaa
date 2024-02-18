import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your dataframe
# Convert timestamp to datetime
df = pd.read_csv('data/sensor.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df=df.drop(['Unnamed: 0','sensor_15'],axis=1)
# df.dropna(axis=1, how='all', inplace=True)
# Fill or interpolate missing values
df = df.interpolate().fillna(method='bfill')

# Combine 'BROKEN' and 'RECOVERING' into a single 'BROKEN' class
df['machine_status'] = df['machine_status'].replace(['BROKEN', 'RECOVERING'], 'BROKEN')
df.to_csv('data/sensor_cleaned.csv', index=False)

print(df.info())