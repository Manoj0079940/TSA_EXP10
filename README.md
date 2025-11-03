# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
# === Import libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# === Load the dataset ===
data = pd.read_csv("AirPassengers.csv")

# Convert 'Month' column to datetime format
data['Month'] = pd.to_datetime(data['Month'], errors='coerce')

# Drop missing or invalid dates
data.dropna(subset=['Month'], inplace=True)

# Sort by date
data = data.sort_values(by='Month')

# Set 'Month' as index
data.set_index('Month', inplace=True)

# Display first few rows
print(data.head())

# === Select the target variable for time series analysis ===
target_col = '#Passengers'

# === Plot the time series ===
plt.figure(figsize=(12, 6))
plt.plot(data.index, data[target_col], label=target_col, color='blue')
plt.xlabel('Month')
plt.ylabel(target_col)
plt.title(f'{target_col} Time Series')
plt.legend()
plt.grid()
plt.show()

# === Function to check stationarity ===
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# === Check stationarity of the series ===
print("\n--- Stationarity Test for #Passengers ---")
check_stationarity(data[target_col])

# === Plot ACF and PACF ===
plt.figure(figsize=(10, 4))
plot_acf(data[target_col].dropna(), lags=30)
plt.title("Autocorrelation Function (ACF)")
plt.show()

plt.figure(figsize=(10, 4))
plot_pacf(data[target_col].dropna(), lags=30)
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

# === Train-Test Split ===
train_size = int(len(data) * 0.8)
train, test = data[target_col][:train_size], data[target_col][train_size:]

# === Build and fit SARIMA model ===
# The seasonal_order=(1,1,1,12) suits monthly data like AirPassengers
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

# === Forecast ===
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# === Evaluate performance ===
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print(f'\nRoot Mean Squared Error (RMSE): {rmse:.4f}')

# === Plot predictions vs actuals ===
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual', color='blue')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.xlabel('Month')
plt.ylabel(target_col)
plt.title(f'SARIMA Model Predictions for {target_col}')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
<img width="216" height="143" alt="image" src="https://github.com/user-attachments/assets/b8e77b33-c24c-47d7-8d17-61cab9fa672d" />
<img width="1119" height="604" alt="image" src="https://github.com/user-attachments/assets/e060ef3d-7e81-44bb-8e0a-66373165354e" />
<img width="364" height="157" alt="image" src="https://github.com/user-attachments/assets/bc74e454-a1a7-43b5-be12-7f57ae4784cf" />
<img width="649" height="501" alt="image" src="https://github.com/user-attachments/assets/31dbfd5c-cec0-4f45-bee8-b80d820cb34f" />
<img width="628" height="487" alt="image" src="https://github.com/user-attachments/assets/ea4b257f-17c4-47bf-ac2e-51be9f3486ef" />
<img width="1115" height="590" alt="image" src="https://github.com/user-attachments/assets/b15f7e8b-78f5-449b-9bea-ec409041cacf" />


### RESULT:
Thus the program run successfully based on the SARIMA model.
