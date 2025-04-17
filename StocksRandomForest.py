
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


# Run the stock prediction function
result = predict_stock_movement("NFLX")

# Show results
print("Accuracy:", result["accuracy"])
print("Classification Report:")
print(result["classification_report"])

