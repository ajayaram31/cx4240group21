import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



raw = pd.read_csv('stock_prices_10yr_all_companies.csv', header=None)

tickers = raw.iloc[0, 1:].values
features = raw.iloc[1, 1:].values
columns = ['Date'] + [f'{tickers[i]}_{features[i]}' for i in range(len(tickers))]

df = pd.DataFrame(raw.values[2:], columns=columns)

# Cleanup
df = df[df['Date'].astype(str) != 'Date']
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
df = df.dropna(subset=['Date'])  # drop rows with invalid dates

# CHANGE THIS TO DIFFERENT TICKER COMPANIES (written in dataLoading.ipynb)
target = 'GOOGL'
cols = [f'Close_{target}', f'High_{target}', f'Low_{target}']
df[cols] = df[cols].astype(float)
df['Prev_Close'] = df[f'Close_{target}'].shift(1)
df.dropna(inplace=True)
# normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Prev_Close'] + cols])
X = scaled[:, 0].reshape(-1, 1)
y = scaled[:, 1:]

split = int(len(X) * 0.9)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
dates_test = df['Date'].iloc[split:]

#one-layer NN
model = Sequential([
    Dense(8, activation='relu', input_shape=(1,)),
    Dense(3)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=30, verbose=0)

# predict
y_pred = model.predict(X_test)
combined_pred = np.hstack([X_test, y_pred])
combined_real = np.hstack([X_test, y_test])
pred_prices = scaler.inverse_transform(combined_pred)
real_prices = scaler.inverse_transform(combined_real)

pred_close = pred_prices[:, 1]
real_close = real_prices[:, 1]
price_diff = pred_close - real_close

threshold = 1.0
signals = np.where(price_diff > threshold, 'Buy',
           np.where(price_diff < -threshold, 'Sell', 'Hold'))

#Trend accuracy
real_trend = np.sign(np.diff(real_close))
pred_trend = np.sign(np.diff(pred_close))
trend_accuracy = accuracy_score(real_trend, pred_trend)

#summary
print(f"Trend prediction accuracy: {trend_accuracy:.2%}")
print("\nSample predictions:")
for i in range(10):
    print(f"{dates_test.iloc[i].date()} | Real: ${real_close[i]:.2f}, Pred: ${pred_close[i]:.2f} | Signal: {signals[i]}")