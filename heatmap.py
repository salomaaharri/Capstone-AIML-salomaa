import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your dataframe
# Convert timestamp to datetime
df = pd.read_csv('data/sensor_cleaned.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Encode machine_status numerically if it's categorical
status_mapping = {'NORMAL': 0, 'BROKEN': 1 }
df['machine_status'] = df['machine_status'].map(status_mapping)

# Calculate correlations
correlations = df.drop(columns=['timestamp']).corrwith(df['machine_status']).drop('machine_status')
print(correlations)

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(correlations.to_frame(), annot=True, cmap='coolwarm')
plt.title('Correlation of Sensors with Machine Status')
plt.show()

broken_sensors = df[df['machine_status'] == 1]
print(broken_sensors)

recover_sensors = df[df['machine_status'] == 2]
print(recover_sensors)

