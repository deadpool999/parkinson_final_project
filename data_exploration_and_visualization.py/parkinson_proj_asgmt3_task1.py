import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# from sklearn.preprocessing import MinMaxScaler

# Read dataset into a DataFrame
df = pd.read_csv("po2_data.csv")

"""
BUILD AND EVALUATE A LINEAR REGRESSION MODEL
"""

# Separate explanatory variables (x) from the response variable (y)
# x = df.iloc[:,:-1].values
# y = df.iloc[:,-1].values


# feature engineering

# df['log_test_time'] = np.log(df['test_time'])
print(df.head())


x = df.drop(['subject#','motor_updrs', 'total_updrs'], axis=1)
y = df[['motor_updrs', 'total_updrs']]


# sns
sns.heatmap(df.corr(), annot = True)
plt.show()

# Split dataset into 60% training and 40% test sets 
# Note: other % split can be used.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Build a linear regression model
model = LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train)

print("Model summary: ")

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({
    "Actual motor_updrs": y_test['motor_updrs'].values,
    "Predicted motor_updrs": y_pred[:, 0],
    "Actual total_updrs": y_test['total_updrs'].values,
    "Predicted total_updrs": y_pred[:, 1]
})

print(df_pred)

# Compute standard performance metrics of the linear regression:

# Compute standard performance metrics for "motor_updrs" prediction
mae_motor = metrics.mean_absolute_error(y_test['motor_updrs'], y_pred[:, 0])
mse_motor = metrics.mean_squared_error(y_test['motor_updrs'], y_pred[:, 0])

# Compute standard performance metrics for "total_updrs" prediction
mae_total = metrics.mean_absolute_error(y_test['total_updrs'], y_pred[:, 1])
mse_total = metrics.mean_squared_error(y_test['total_updrs'], y_pred[:, 1])

print("Motor UPDRS performance:")
print("MAE: ", mae_motor)
print("MSE: ", mse_motor)

print("Total UPDRS performance:")
print("MAE: ", mae_total)
print("MSE: ", mse_total)




# Compute RMSE for "motor_updrs" prediction
rmse_motor = math.sqrt(metrics.mean_squared_error(y_test['motor_updrs'], y_pred[:, 0]))

# Compute RMSE for "total_updrs" prediction
rmse_total = math.sqrt(metrics.mean_squared_error(y_test['total_updrs'], y_pred[:, 1]))

# Compute NRMSE for "motor_updrs" prediction
y_max_motor = y_test['motor_updrs'].max()
y_min_motor = y_test['motor_updrs'].min()
nrmse_motor = rmse_motor / (y_max_motor - y_min_motor)

# Compute NRMSE for "total_updrs" prediction
y_max_total = y_test['total_updrs'].max()
y_min_total = y_test['total_updrs'].min()
nrmse_total = rmse_total / (y_max_total - y_min_total)

# Compute R^2 for "motor_updrs" prediction
r2_motor = metrics.r2_score(y_test['motor_updrs'], y_pred[:, 0])

# Compute R^2 for "total_updrs" prediction
r2_total = metrics.r2_score(y_test['total_updrs'], y_pred[:, 1])

print("Motor UPDRS performance:")
print("RMSE: ", rmse_motor)
print("NRMSE: ", nrmse_motor)
print("R^2: ", r2_motor)

print("Total UPDRS performance:")
print("RMSE: ", rmse_total)
print("NRMSE: ", nrmse_total)
print("R^2: ", r2_total)

