import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read dataset into a DataFrame
df = pd.read_csv("po2_data.csv")

# Separate explanatory variables (x) from the response variable (y)
x = df.drop(['subject#', 'motor_updrs', 'total_updrs'], axis=1)
y = df[['motor_updrs', 'total_updrs']]

# Define a list of different test sizes for train-test splits
test_sizes = [0.5, 0.4, 0.3, 0.2]

# Create lists to store evaluation metrics for each scenario
mae_motor_list = []
mse_motor_list = []
mae_total_list = []
mse_total_list = []
r2_motor_list = []  # Added
r2_total_list = []  # Added

for test_size in test_sizes:
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=2)

    # Build a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Use linear regression to predict the values of (y) in the test set
    y_pred = model.predict(X_test)

    # Compute MAE, MSE, and R-squared for "motor_updrs" and "total_updrs" predictions
    mae_motor = metrics.mean_absolute_error(y_test['motor_updrs'], y_pred[:, 0])
    mse_motor = metrics.mean_squared_error(y_test['motor_updrs'], y_pred[:, 0])
    r2_motor = metrics.r2_score(y_test['motor_updrs'], y_pred[:, 0])
    
    mae_total = metrics.mean_absolute_error(y_test['total_updrs'], y_pred[:, 1])
    mse_total = metrics.mean_squared_error(y_test['total_updrs'], y_pred[:, 1])
    r2_total = metrics.r2_score(y_test['total_updrs'], y_pred[:, 1])

    # Append metrics to the respective lists
    mae_motor_list.append(mae_motor)
    mse_motor_list.append(mse_motor)
    r2_motor_list.append(r2_motor)
    mae_total_list.append(mae_total)
    mse_total_list.append(mse_total)
    r2_total_list.append(r2_total)

# Print performance metrics for each scenario
for i, test_size in enumerate(test_sizes):
    print(f"Test Size {int(test_size * 100)}%:")
    print("Motor UPDRS performance:")
    print("MAE: ", mae_motor_list[i])
    print("MSE: ", mse_motor_list[i])
    print("R^2: ", r2_motor_list[i])
    print("Total UPDRS performance:")
    print("MAE: ", mae_total_list[i])
    print("MSE: ", mse_total_list[i])
    print("R^2: ", r2_total_list[i])
    print("\n")

# Plot performance metrics
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(test_sizes, mae_motor_list, marker='o', label='Motor UPDRS MAE')
plt.plot(test_sizes, mae_total_list, marker='o', label='Total UPDRS MAE')
plt.xlabel('Test Size')
plt.ylabel('MAE')
plt.title('MAE vs. Test Size')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(test_sizes, r2_motor_list, marker='o', label='Motor UPDRS R^2')
plt.plot(test_sizes, r2_total_list, marker='o', label='Total UPDRS R^2')
plt.xlabel('Test Size')
plt.ylabel('R^2')
plt.title('R^2 vs. Test Size')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
