import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import featurex

data = pd.read_csv('data/sensor_cleaned.csv')

data = data[featurex.columns]
status_mapping = {'NORMAL': 0, 'BROKEN': 1 }
data['machine_status'] = data['machine_status'].map(status_mapping)

# Assuming 'data' is your DataFrame
numeric_data = data.select_dtypes(include=[np.number])
correlations = numeric_data.corr()

# Focus on correlations with 'machine_status'
machine_status_corr = correlations['machine_status'].sort_values(ascending=False)

# Print or view the correlation values
print(machine_status_corr)

# Plotting the correlations can be helpful
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.show()

# Convert data to format suitable for sklearn models
# Assuming 'sensor_columns' contains relevant sensor names
X = data[featurex.sensors]
y = data['machine_status']

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importance
importances = model.feature_importances_

# Plot feature importance
plt.barh(featurex.sensors, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Sensor")
plt.show()


mi = mutual_info_classif(X, y)

# Plot mutual information
plt.barh(featurex.sensors, mi)
plt.xlabel("Mutual Information")
plt.ylabel("Sensor")
plt.show()