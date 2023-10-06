import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("po2_data.csv")

# Pairplot to visualize relationships between variables
sns.pairplot(data, vars=['motor_updrs', 'total_updrs', 'age', 'jitter(%)', 'shimmer(%)'], hue='sex')
plt.suptitle("Pairplot of Variables", y=1.02)
plt.show()

# Correlation matrix heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Scatter plot: motor_updrs vs. age
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='motor_updrs', data=data)
plt.title('Motor UPDRS Score vs. Age')
plt.xlabel('Age')
plt.ylabel('Motor UPDRS Score')
plt.show()

# Scatter plot: total_updrs vs. age
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='total_updrs', data=data)
plt.title('Total UPDRS Score vs. Age')
plt.xlabel('Age')
plt.ylabel('Total UPDRS Score')
plt.show()
