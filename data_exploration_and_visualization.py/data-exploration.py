import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("po2_data.csv")

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Summary statistics for numeric columns
print("\nSummary Statistics:")
print(data.describe())

# Data visualization
plt.figure(figsize=(12, 6))

# Histograms for motor_updrs and total_updrs scores
plt.subplot(2, 2, 1)
sns.histplot(data['motor_updrs'], bins=20, kde=True)
plt.title('Motor UPDRS Score Distribution')

plt.subplot(2, 2, 2)
sns.histplot(data['total_updrs'], bins=20, kde=True)
plt.title('Total UPDRS Score Distribution')

# Scatter plot for test_time vs. motor_updrs
plt.subplot(2, 2, 3)
sns.scatterplot(x='test_time', y='motor_updrs', data=data)
plt.title('test_time vs. Motor UPDRS Score')

# Scatter plot for test-time vs. total_updrs
plt.subplot(2, 2, 4)
sns.scatterplot(x='test_time', y='total_updrs', data=data)
plt.title('test_time vs. Total UPDRS Score')

plt.tight_layout()
plt.show()
