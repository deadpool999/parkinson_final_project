import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import PowerTransformer
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with the actual filename)
df = pd.read_csv('po2_data.csv')

# Feature engineering: Creating new features
df['feature_new1'] = df['shimmer(apq11)'] - df['shimmer(apq3)']  # Example: Difference between 'shimmer(apq3)' and 'shimmer(apq11)'
df['feature_new2'] = df['shimmer(apq5)'] - df['shimmer(apq3)'] 
df['feature_new3'] = df['jitter(abs)'] * df['jitter(%)']  
df['feature_new4'] = df['jitter(%)'] * df['nhr'] 
df['feature_new5'] = df['shimmer(%)'] * df['shimmer(apq5)'] 
df['feature_new6'] = df['jitter(%)'] * df['ppe']
df['feature_new7'] = df['jitter(%)'] * df['jitter(abs)']

df['feature_new8'] = df['jitter(ddp)'] * df['jitter(abs)']
df['feature_new9'] = df['jitter(ddp)'] * df['jitter(ppq5)']
df['feature_new10'] = df['shimmer(apq11)'] * df['shimmer(dda)']
df['feature_new11'] = df['jitter(%)'] * df['shimmer(abs)']

# ---------------------            --------------------------                      ---------------------
# ---------------------            --------------------------                      ---------------------
# Apply non-linear transformation
# ---------------------            --------------------------                      ---------------------
# ---------------------            --------------------------                      ---------------------

for key_var in df.columns:
    if not (key_var == 'subject#' or key_var == 'motor_updrs' or key_var== 'sex' or key_var == 'test_time'or key_var ==  'total_updrs' or key_var == 'feature_new1' or key_var == 'feature_new2'):
     
        sample = df[key_var].to_numpy()
        df["Log_" + key_var] = df[key_var].apply(np.log)
        df = df.drop(key_var, axis=1)
        
# df = df[['Log_age','test_time','motor_updrs','total_updrs','Log_jitter(%)','Log_jitter(abs)','Log_jitter(rap)','Log_jitter(ppq5)','Log_jitter(ddp)','Log_shimmer(%)','Log_shimmer(abs)','Log_shimmer(apq3)','Log_shimmer(apq5)','Log_shimmer(apq11)','Log_shimmer(dda)', 'Log_nhr', 'Log_hnr', 'Log_rpde', 'Log_dfa','Log_ppe']]
print(df.head())
# checking if any columns has missing values
print(df.isnull().sum())


# Creating a histogram to know the distributions of data after applying log transformation
for key_var in df.columns:
    if key_var != 'subject#':
        sample = df[key_var].to_numpy()
        # Creating a histogram to know the distributions of data
        plt.figure(figsize=(8, 6))
        plt.hist(sample, bins=20, alpha=0.5, label='data')
        plt.title(f"Histogram of {key_var}")
        plt.xlabel(key_var)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


# Split the dataset into features (X) and target (y)
X = df.drop(columns=['subject#', 'motor_updrs', 'total_updrs'])  # Features
y_motor = df['motor_updrs']  # Target: Motor UPDRS
y_total = df['total_updrs']  # Target: Total UPDRS

# Split the data into training and testing sets
# X_train, X_test, y_train_motor, y_test_motor, y_train_total, y_test_total = train_test_split(
#     X, y_motor, y_total, test_size=0.2, random_state=2
# )

# ---------------------            --------------------------                      ---------------------
# ---------------------            --------------------------                      ---------------------
# Apply standardization
# ---------------------            --------------------------                      ---------------------
# ---------------------            --------------------------                      ---------------------
# Applying StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------            --------------------------                      ---------------------
# ---------------------            --------------------------                      ---------------------
# Apply Gausssian Transformation
# ---------------------            --------------------------                      ---------------------
# ---------------------            --------------------------                      ---------------------

# Applying StandardScaler
# Standardize the features using the Yeo-Johnson transformation
scaler = PowerTransformer(method='yeo-johnson')
X_transformed = scaler.fit_transform(X_scaled)

# Split the data into training and testing sets for Gaussian-transformed features
X_train, X_test,y_train_motor, y_test_motor, y_train_total, y_test_total  = train_test_split(
    X_transformed, y_motor, y_total, test_size=0.2, random_state=2
)

# Create and train your linear regression models
model_motor = LinearRegression()
model_motor.fit(X_train, y_train_motor)

model_total = LinearRegression()
model_total.fit(X_train, y_train_total)

# Make predictions
y_pred_motor = model_motor.predict(X_test)
y_pred_total = model_total.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_motor = mean_squared_error(y_test_motor, y_pred_motor)
mse_total = mean_squared_error(y_test_total, y_pred_total)

# Calculate Mean Absolute Error (MAE)
mae_motor = mean_absolute_error(y_test_motor, y_pred_motor)
mae_total = mean_absolute_error(y_test_total, y_pred_total)

# Calculate Root Mean Squared Error (RMSE)
rmse_motor = math.sqrt(mean_squared_error(y_test_motor, y_pred_motor))
rmse_total = math.sqrt(mean_squared_error(y_test_total, y_pred_total))

# Compute NRMSE for "motor_updrs" prediction
y_max_motor = y_test_motor.max()
y_min_motor = y_test_motor.min()
nrmse_motor = rmse_motor / (y_max_motor - y_min_motor)

# Compute NRMSE for "total_updrs" prediction
y_max_total = y_test_total.max()
y_min_total = y_test_total.min()
nrmse_total = rmse_total / (y_max_total - y_min_total)


# Calculate R-squared
r2_motor = r2_score(y_test_motor, y_pred_motor)
r2_total = r2_score(y_test_total, y_pred_total)

# Calculate Adjusted R-squared
n_samples, n_features = X_test.shape
adj_r2_motor = 1 - (1 - r2_motor) * ((n_samples - 1) / (n_samples - n_features - 1))
adj_r2_total = 1 - (1 - r2_total) * ((n_samples - 1) / (n_samples - n_features - 1))

# Print the results
print("Model Performance for Motor UPDRS:")
print(f"Mean Absolute Error: {mae_motor}") 
print(f"Root Mean Squared Error: {rmse_motor}")  
print(f"Normalized RMSE: {nrmse_motor}") 

print(f"Mean Squared Error: {mse_motor}")
print(f"R-squared: {r2_motor}")
print(f"Adjusted R-squared for Motor UPDRS: {adj_r2_motor}\n")

print("Model Performance for Total UPDRS:")
print(f"Mean Absolute Error: {mae_total}")  
print(f"Root Mean Squared Error: {rmse_total}") 
print(f"Normalized RMSE: {nrmse_total}")  

print(f"Mean Squared Error: {mse_total}")
print(f"R-squared: {r2_total}")
print(f"Adjusted R-squared for Total UPDRS: {adj_r2_total}")