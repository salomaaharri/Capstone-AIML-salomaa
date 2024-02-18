from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import featurex

# Assuming 'data' is your DataFrame
data = pd.read_csv('data/sensor_cleaned.csv')

# Convert timestamp to datetime
# data['timestamp'] = pd.to_datetime(data['timestamp'])

# Assuming you have a DataFrame 'data' with sensor readings
X = data[featurex.sensors]  # sensor_columns should contain only sensor feature columns

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize Isolation Forest
iso_forest = IsolationForest(contamination=0.01)  # contamination is an estimate of the proportion of anomalies

# Train the model
iso_forest.fit(X_scaled)

# Predict anomalies
anomalies = iso_forest.predict(X_scaled)
data['anomaly'] = anomalies

# Anomalies will be marked with -1, normal data with 1
data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})

print(data['anomaly'].head())


# Count the number of anomalies
anomaly_count = data['anomaly'].value_counts()

# Assuming anomaly_count is your series with the counts
print("Anomaly Counts")
print(anomaly_count)

# Print the counts for normal data points and anomalies
print("\nNumber of Normal Data Points:", anomaly_count[0])  # Normal points are marked as 0
print("Number of Anomalies:", anomaly_count[1])  # Anomalies are marked as 1


