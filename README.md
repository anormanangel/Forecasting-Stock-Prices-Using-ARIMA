# Forecasting Stock Prices Using ARIMA

# Introduction
This project focuses on forecasting stock prices using the ARIMA (AutoRegressive Integrated Moving Average) model. Accurate stock price predictions can provide valuable insights for investors, enabling them to make informed decisions and potentially maximize their returns.

## Autoregressive Integrated Moving Average (ARIMA)
An autoregressive integrated moving average, or ARIMA, is a statistical analysis model that uses time series data to predict future trends. it predicts future values based on past values

### ARIMA Parameters
For ARIMA models, a standard notation would be ARIMA with p, d, and q, where integer values substitute for the parameters to 
indicate the type of ARIMA model used. The parameters can be defined as:

* p: the number of lag observations in the model, also known as the lag order.
* d: the number of times the raw observations are differenced; also known as the degree of differencing.
* q: the size of the moving average window, also known as the order of the moving average.

Python code? Check them out here on GitHub: 

# Background
Stock price forecasting is a critical aspect of financial analysis, providing investors with predictions on future price movements. Traditional methods often struggle to account for the inherent volatility and noise in stock price data. The ARIMA model, however, offers a robust statistical approach that can handle such complexities. By leveraging historical price data, the ARIMA model aims to identify underlying patterns and make future price predictions, thus aiding investors in their decision-making processes.

The project entails loading historical stock price data, preprocessing it to ensure accuracy, fitting an ARIMA model to the data, and using the model to forecast future prices. The efficacy of the model is evaluated, and recommendations are made based on the forecasts. ​retain customers.

# The Dataset
The dataset used in this project comprises real-time stock price data obtained from Yahoo Finance, which typically includes the following attributes:

1. Date: The specific trading day.
2. Open: The price at which the stock opened on a particular day.
3. High: The highest price reached by the stock during the trading day.
4. Low: The lowest price reached by the stock during the trading day.
5. Close: The price at which the stock closed at the end of the trading day.
6. Volume: The number of shares traded during the day.
7. Adjusted Close: The closing price is adjusted for corporate actions such as dividends, stock splits, etc.

Do you want to explore real-time stock data from Yahoo Finance? Check it out: https://finance.yahoo.com/

### The three Questions I was asking were;

1. How effective is the ARIMA model in predicting future stock prices based on historical data?

2. What is the potential ROI when using ARIMA model forecasts for stock trading compared to a traditional buy-and-hold strategy?

3. How does removing volatility during data preprocessing impact the accuracy of stock price forecasts using the ARIMA model?

# Tools I Used
To forecast stock prices and implement ARIMA and different trading strategies, I used the powers of several key tools & Libraries

1. **Libraries:**
    * pandas: For data manipulation and analysis.
    * numpy: For numerical operations.
    * matplotlib and seaborn: For data visualization.
    * statsmodels: For statistical modeling, particularly for conducting statistical data exploration and performing statistical tests. It is built on top of NumPy, SciPy, and matplotlib,
2. **ARIMA Model:** Used for forecasting stock prices based on historical data.

# Time Series Analysis: EDA & Preprocessing for ARIMA Modeling

### 1. Autocorrelation.
The autocorrelation plot for Apple stock with lag 3 shows a strong positive linear relationship. This tight, diagonal clustering indicates high autocorrelation, suggesting recent past values strongly influence current prices. This pattern supports using an ARIMA model and implies that incorporating a lag of 3 could improve forecasting accuracy.

![Autocorrelation plot with a Lag of 3](https://github.com/anormanangel/Forecasting-Stock-Prices-Using-ARIMA/blob/main/Assets/Cross%20Correlation.png)
*Scatter Plot showing the correlation of a time series (Stick Price) with a lagged version of itself*

### 2. Stock Price Evolution Over Time
The Apple stock price chart shows a downward trend from January 2022 to March 2023, with high volatility. Notable peaks and troughs are visible, including a significant drop to the lowest point around January 2023, followed by a slight recovery. The non-stationary pattern suggests differencing may be necessary for ARIMA modeling.

![Stock Price Evolution Over Time](https://github.com/anormanangel/Forecasting-Stock-Prices-Using-ARIMA/blob/main/Assets/Stock%20Price%20Evolution%20Over%20Time.png)
*Bar chat showing all columns in the dataset with no missing data*

### 3. Augmented Dickey-Fuller (ADF) Test to Check for Stationarity.
The results from ADF These results are from an Augmented Dickey-Fuller (ADF) test showed test statistic (-2.716453) is greater than the 5% critical value (-2.871), and the p-value (0.071234) is above 0.05. This indicates that we fail to reject the null hypothesis of non-stationarity at the 5% significance level. Therefore, the time series is not stationary and will likely require differencing for ARIMA modeling.

1. **Test Statistic:** -2.716453
2. **p-value: 0.071234**
3. **Critical Values:**
	
        1%: -3.453

	    5%: -2.871

	    10%: -2.572

The time series is not stationary.

### 4. Differencng to Remove Trend & Seasonality.
The differenced Apple stock price plot shows fluctuations around zero, indicating improved stationarity. The data exhibits no clear trend, with price changes varying between approximately -10 and +10 units. Volatility appears consistent throughout, suggesting the differencing has addressed the non-stationarity issue observed in the original time series. This transformed data is likely more suitable for ARIMA modeling.

![Differenced Stock Price](https://github.com/anormanangel/Forecasting-Stock-Prices-Using-ARIMA/blob/main/Assets/Differenced%20Time%20Series.png)
*A plot showing the differenced stock price (Stationary data)*

### 5. Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)

The ACF and PACF plots show:
* ACF: Sharp drop after lag 0, with subsequent lags mostly within confidence bounds.
* PACF: Significant spike at lag 1, then mostly within bounds.

This pattern suggests an ARIMA(1,1,0) or ARIMA(0,1,1) model may be appropriate. The single significant lag in PACF indicates an AR(1) component, while the quick drop in ACF could suggest an MA(1) term. The differencing applied earlier accounts for the integrated (I) component.

![ACF & PACF Plots](https://github.com/anormanangel/Forecasting-Stock-Prices-Using-ARIMA/blob/main/Assets/Auto%20and%20Partial%20Autocorrelation.png)
*ACF and PACF Plots*

# Modeling Using ARIMA

### Finding the Best ARIMA Model

In this section, we aimed to identify the optimal ARIMA model by evaluating all combinations of specified 𝑝, 𝑑, and 𝑞 values and selecting the one that minimized the Akaike Information Criterion (AIC). We defined a range for:
* 𝑝 (0 to 2), 
* 𝑑 (0 to 1), and 
* 𝑞 (0 to 2) 

Then all possible combinations of these parameters. For each combination, an ARIMA model was fitted to the data, and its AIC was computed. The model with the lowest AIC value was selected as the best model. 

 **Critical Values:**
	
    The optimal parameters
    𝑝 = 1 
    𝑑 = 1, and 
    𝑞 = 2 

### Fitting the ARIMA Model with Optimal Parameters
We fitted the ARIMA model with the optimal parameters 𝑝 = 1, 𝑑= 1, and 𝑞 = 2 q=2 obtained previously. The model was fitted to the differenced data, and a summary of the fitted model was printed. 

The summary revealed that the ARIMA(1, 1, 2) model had a log-likelihood of -757.338, an AIC of 1522.677, and a BIC of 1537.370. The coefficients for the AR term (ar.L1) were -0.8980, while the MA terms (ma.L1 and ma.L2) were -0.0205 and -0.9793, respectively. The sigma^2 (variance of residuals) was 10.4433. Residual diagnostics indicated a Ljung-Box Q statistic of 1.16 (p-value = 0.28) and a Jarque-Bera test statistic of 16.28 (p-value = 0.00), suggesting some non-normality in the residuals. The residuals were plotted to visually assess any remaining patterns or issues.

The SARIMAX results provide valuable insights into the ARIMA(1,1,2) model fitted to the stock price data. The Ljung-Box test (Q) with a p-value of 0.28 suggests no significant autocorrelation in the residuals, indicating a good model fit. However, the Jarque-Bera test's low p-value (0.00) implies non-normality in the residuals, which is common in financial data. The heteroskedasticity test (p-value 0.23) indicates constant variance in the residuals. The kurtosis of 4.16 confirms the non-normality, showing heavier tails than a normal distribution. These diagnostics suggest that while the ARIMA model captures the time series dynamics well

![SARIMAX Results of ARIMA 1,1,2](https://github.com/anormanangel/Forecasting-Stock-Prices-Using-ARIMA/blob/main/Assets/SARIMAX%20Results%20of%20ARIMA%201%2C1%2C2.png)
*SARIMAX Results of ARIMA 1,1,2*

### ARIMA Model Forecast
The red line at the end of the series represents the ARIMA model's forecast. It suggests a slight downward trend in the immediate future, though the forecast appears quite short-term. The model predicts a small decrease in price over the next few time periods.

![ARIMA Model Forecast](https://github.com/anormanangel/Forecasting-Stock-Prices-Using-ARIMA/blob/main/Assets/ARIMA%20Model%20Forecast.png)
*ARIMA Model Forecast*

### Calculating Returns Using the Buy and Hold Strategy
In this section, the returns from a buy-and-hold strategy were calculated. The initial price was taken as the closing price on the first day, and the final price was the closing price on the last day. The return was computed as
(final_price−initial_price)/initial_price, which resulted in a return of -17.02%. The last day's closing price was $151.03.

### Forecasting Prices for the Next 10 Days
Using the ARIMA model, we forecasted the prices for the next 10 days. The forecasted prices, adjusted by adding the last day's closing price of $151.03, are as follows:

1. $150.62
2. $151.19
3. $150.68
4. $151.14
5. $150.73
6. $151.10
7. $150.76
8. $151.06
9. $150.80
10. $151.04

These forecasts provide an estimate of future prices based on the model's projections.

# Insights & Recommendations
1. Buying and holding an investment is a common approach, but it might not always yield the highest returns. For example, in this case, if we had bought and held the investment from the first day until today, we would have lost 17.02% of our investment.

2. However, the ARIMA model offers a way to predict the ROI of the investment in 2 days, which is forecasted to be 0.11%. This is a small but positive return, and it suggests that the ARIMA model might be a better approach to investment than buying and holding.

3. It's important to keep in mind that the ARIMA model's predictions aren't always spot-on, and many factors can affect an investment's price, such as market trends, news events, and global economic conditions. Nonetheless, the ARIMA model can still be a useful tool to assist us in making informed investment decisions.

# What I Learned
* 🧩 **Advanced Python Concept for Data Analytics:** Mastered the art of using Python for data manipulation, processing, and visualization of Time Series datasets including  stock price evolution over time, Augmented Dickey-Fuller (ADF) Test to Check for Stationarity, differencing to remove seasonality and trend, and autocorrelation

* 📊 **Time Series Modeling:** I implemented ARIMA model first by finding the best ARIMA, I selected ARIMA 1,1,2 then fit and forecasted stock prices 

* 📌 **Calculation Return in Investment** I learned how to calculate return on investment using the buy and Hold Strategy and compared it with the ARIMA forecast

* 💡 **Analytical Wizardry:** Leveled up my real-world problem-solving skills turning questions into actionable insights using Python

# Conclusions

In this project Implementing the ARIMA model to forecast future stock prices. Additionally, I calculated the return on investment using the Buy and Hold strategy and compared it to the ARIMA forecast, providing a practical perspective on investment strategies, such as Buy and Hold versus ARIMA forecast, which helps in choosing the most effective investment approach.


