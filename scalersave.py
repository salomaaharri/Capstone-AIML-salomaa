from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import pandas as pd
import featurex

# Load your data
data = pd.read_csv('data/sensor_engineered.csv')

# Initialize and fit the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data[featurex.sensors])  # Only scale the sensor columns

# Save the fitted scaler
dump(scaler, 'data/scaler.joblib')
