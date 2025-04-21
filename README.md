# cx4240group21
Download all files from Github.
Open dataLoading.ipynb
If you want to change the year or frequency of stock closing prices, go to the following line in this file and change the period from "10y" to another period:
data = yf.download(tickers, period="10y")["Close"]
Run dataLoading.ipynb

# NEURAL NETWORK
If not installed yet, pip install tensorflow and other imports used in nn.py
If using 3.13 python or above, use this instead in a powershell command prompt: pip install tf-nightly
Change the target label on line 25 from 'GOOGL' to other targets listed in dataLoading.ipynb to view predictions for other companies in the data frame

# RANDOM FOREST
Run StocksRandomForest.py
Change the target label on line 100 to different targets in the dataframe (listed in dataLoading.ipynb) to see results for different companies' data. 
The output will give you the accuracy of the model, and a table of the last 5 trading days. 
To plot the model's predictions change the target label on line 125 to a different target in the dataframe.  

# LOGSTIC REGRESSION
Go to https://fred.stlouisfed.org/series/UNRATE and download a 10 year csv file for the unemployment rate. Name this file UNRATE.csv
Run all three cells in logistic_regression.ipynb

# LINEAR REGRESSION
Run CX4240 Final Project Linear Regression. Change the ticker label on line 11 in the 3rd cell to different ticker labels to see predictions and actual values for the companies used in dataLoading.ipynb. 
