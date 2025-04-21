
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = pd.Series(preds, index=test.index)
    preds_binary = (preds > 0.6).astype(int)  # Threshold can be adjusted
    return pd.DataFrame({
        "Predictions": preds_binary,
        "Target": test["Target"]
    }, index=test.index)

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


def predict_stock_movement(ticker_symbol, n_estimators=200, min_samples_split=100, test_size=100):
    # Step 1: Fetch historical data
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period="max")
    
    # Step 2: Create target column (1 if price went up next day, else 0)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.dropna()

    # Step 3: Features and formatting
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    data.index = pd.to_datetime(data.index)

    # Step 4: Split data into train/test
    train = data.iloc[:-test_size]
    test = data.iloc[-test_size:]

    # Step 5: Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        random_state=1
    )
    model.fit(train[predictors], train["Target"])

    # Step 6: Evaluate model
    predictions = backtest(data, model, predictors)
    accuracy = accuracy_score(predictions["Target"], predictions["Predictions"])
    report = classification_report(predictions["Target"], predictions["Predictions"], output_dict=True)

    return {
        "model": model,
        "accuracy": accuracy,
        "classification_report": report,
        "predictions": predictions
    }

def generate_trade_signals(predictions_df):
    decisions = predictions_df.copy()
    decisions["Signal"] = decisions["Predictions"].apply(lambda x: "Buy" if x == 1 else "Sell/Hold")
    return decisions[["Predictions", "Target", "Signal"]]

import matplotlib.pyplot as plt

def plot_predictions(predictions_df, stock_data, signal_interval=5):
    signals = predictions_df.copy()
    signals["Close"] = stock_data.loc[signals.index, "Close"]

    plt.figure(figsize=(14, 6))
    plt.plot(signals.index, signals["Close"], label="Stock Price", linewidth=2)

    # Sample every Nth signal for clarity
    buy_signals = signals[signals["Predictions"] == 1].iloc[::signal_interval]
    sell_signals = signals[signals["Predictions"] == 0].iloc[::signal_interval]

    # Plot sampled Buy/Sell signals
    plt.scatter(buy_signals.index, buy_signals["Close"], label="Buy Signal", marker="^", color="green", s=100)
    plt.scatter(sell_signals.index, sell_signals["Close"], label="Sell/Hold Signal", marker="v", color="red", s=100)

    plt.title("Stock Price with Buy/Sell Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Run the stock prediction function, displays accuracy and shows results for last 5 signals
result = predict_stock_movement("META")
print("Accuracy META:", result["accuracy"])
trade_signals = generate_trade_signals(result["predictions"])
print("\nRecent Trade Signals:")
print(trade_signals.tail(5))
print("\n")
result = predict_stock_movement("GOOGL")
print("Accuracy GOOGL:", result["accuracy"])
trade_signals = generate_trade_signals(result["predictions"])
print("\nRecent Trade Signals:")
print(trade_signals.tail(5))
print("\n")
result = predict_stock_movement("VZ")
print("Accuracy VZ:", result["accuracy"])
trade_signals = generate_trade_signals(result["predictions"])
print("\nRecent Trade Signals:")
print(trade_signals.tail(5))
print("\n")
result = predict_stock_movement("NFLX")
print("Accuracy NFLX:", result["accuracy"])
trade_signals = generate_trade_signals(result["predictions"])
print("\nRecent Trade Signals:")
print(trade_signals.tail(5))

# Plot with less clutter: show only every 7th signal
plot_predictions(result["predictions"], yf.Ticker("NFLX").history(period="max"), signal_interval=7)








