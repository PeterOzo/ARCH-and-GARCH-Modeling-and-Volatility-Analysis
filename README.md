## ARCH/GARCH Modeling & Volatility Analysis
*FIN 666 - Advanced Quantitative Methods and Machine Learning in Finance*

## **Business Question**
How can financial analysts and risk managers effectively model time-varying volatility in financial markets using ARCH/GARCH frameworks to enhance risk assessment, portfolio optimization, and derivative pricing for both retail sales forecasting and equity index volatility prediction?

## **Business Case**
In modern financial markets, accurate volatility modeling is essential for risk management, derivative pricing, and portfolio optimization. Traditional time series models like ARMA assume constant variance (homoskedasticity), which fails to capture the reality of financial data where volatility clusters in periods of market stress. ARCH (Autoregressive Conditional Heteroskedasticity) and GARCH (Generalized ARCH) models address this critical limitation by modeling time-varying conditional variance. This capability is crucial for financial institutions to calculate Value-at-Risk (VaR), optimize trading strategies, price options accurately, and meet regulatory capital requirements. By implementing sophisticated volatility models, institutions can better understand market dynamics, improve risk-adjusted returns, and develop more effective hedging strategies during periods of market turbulence.

## **Analytics Question**
How can the systematic application of ARCH/GARCH modeling techniques, combined with comprehensive diagnostic testing and model selection procedures, help financial analysts develop robust volatility forecasting models that capture heteroskedasticity patterns in both retail sales data and equity returns while ensuring model adequacy through rigorous residual analysis?

## **Outcome Variables of Interest**
The analysis focuses on two primary outcome variables:
1. **Advance Retail Sales (RSFHFS)**: Monthly furniture and home furnishings store sales for forecasting business cycles
2. **DJIA Daily Returns**: Percentage returns capturing equity market volatility dynamics and risk characteristics

## **Key Predictors**
The ARCH/GARCH framework employs conditional variance predictors:
- **ARCH Terms (α)**: Past squared error terms capturing volatility shocks
- **GARCH Terms (β)**: Lagged conditional variance terms modeling volatility persistence  
- **Constant Term (ω)**: Long-run unconditional variance component
- **Mean Equation**: Constant or ARMA components for return modeling
- **Volatility Clustering Effects**: Time-varying conditional heteroskedasticity patterns

## **Dataset Description**

### **Retail Sales Dataset (RSFHFS)**
The Advance Retail Sales dataset contains monthly seasonally-adjusted sales figures for Furniture and Home Furnishings Stores, sourced from the Federal Reserve Economic Data (FRED) database. This dataset provides crucial insights into consumer spending patterns and economic cycles.

**Dataset Specifications:**
- **Temporal Coverage**: January 2000 to December 2019 (240 monthly observations)
- **Frequency**: Monthly, seasonally adjusted
- **Variable**: Sales in millions USD
- **Data Source**: [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/RSFHFS)
- **Economic Significance**: Leading indicator of consumer spending and economic health

### **DJIA Returns Dataset**
The Dow Jones Industrial Average dataset contains daily closing values for comprehensive volatility analysis of major U.S. equity market movements.

**Dataset Specifications:**
- **Temporal Coverage**: Past 5 years (2020-2025, approximately 1,302 observations)
- **Frequency**: Daily trading data
- **Variable**: DJIA index levels and computed daily percentage returns
- **Data Source**: [Yahoo Finance Historical Data](https://finance.yahoo.com/)
- **Market Significance**: Benchmark index representing 30 large-cap U.S. companies

## **Theoretical Framework & ARCH/GARCH Foundations**

### **When to Use ARCH/GARCH Models**
ARCH family models are essential when time series exhibit specific characteristics that violate ARMA assumptions:

**Key Situations Requiring ARCH Models:**
1. **Volatility Clustering**: Periods of high volatility followed by calm periods
2. **Non-Constant Variance**: Heteroskedasticity where error variance changes over time
3. **Autocorrelated Squared Residuals**: ARMA residuals showing correlation in squared terms
4. **Financial Risk Modeling**: VaR calculations and derivative pricing applications
5. **Macroeconomic Volatility**: Exchange rates, inflation, and energy price modeling

### **Diagnostic Framework for ARCH Effects**
**Step 1: ARMA Model Fitting and Residual Analysis**
- Fit optimal ARMA model and extract residuals
- Test residual stationarity using ADF test
- Verify white noise properties through ACF analysis

**Step 2: Heteroskedasticity Detection**
- Visual inspection for volatility clustering patterns
- ACF and PACF analysis of squared residuals
- Identification of time-varying variance structure

**Step 3: Statistical Testing for ARCH Effects**
- **Engle's ARCH Test**: Tests null hypothesis of constant variance
- **Ljung-Box Test**: Examines autocorrelation in squared residuals
- **Critical Thresholds**: p-value < 0.05 indicates ARCH effects

**Step 4: ARCH Order Determination**
- **PACF Analysis**: Significant lags indicate optimal ARCH order
- **Information Criteria**: AIC/BIC comparison across different orders
- **Rule of Thumb**: Single PACF spike suggests ARCH(1), multiple spikes indicate higher orders

## **Data Analysis & Preprocessing**

### **RSFHFS Time Series Analysis**

![image](https://github.com/user-attachments/assets/86b55ac6-d6b1-440b-89a4-5daf34acfb7a)


**Stationarity Assessment:**
- **Original Series**: ADF p-value = 0.5707 (non-stationary)
- **First Difference**: ADF p-value = 0.0627 (marginally stationary)
- **Conclusion**: ARIMA modeling required due to trending behavior

**Key Statistical Properties:**
- **Mean**: $8,456.72 million (2000-2019 average)
- **Standard Deviation**: $945.80 million
- **Range**: $6,915 - $10,252 million
- **Trend**: Clear upward trajectory with crisis-related volatility (2008-2009)

### **DJIA Returns Analysis**

![image](https://github.com/user-attachments/assets/fa0d10e7-7d23-4a8e-94e9-d8465266d3fc)


![image](https://github.com/user-attachments/assets/055ee694-0339-49fc-bb7e-616e0244bad9)


**Return Characteristics:**
- **Mean Daily Return**: 0.053% (13.7% annualized)
- **Daily Volatility**: 1.22% (20.6% annualized)
- **Return Range**: -12.93% to +11.37%
- **Distribution**: Fat tails with excess kurtosis typical of financial returns

**Volatility Clustering Evidence:**
- Clear periods of high/low volatility clustering
- Particularly evident during 2020 COVID-19 market disruption
- Squared returns show persistent autocorrelation patterns

## **Model Selection & Implementation**

### **ARIMA Model for RSFHFS**

**Comprehensive Model Evaluation:**
| **Model** | **AIC** | **BIC** | **Log Likelihood** | **Selection Rank** |
|-----------|---------|---------|-------------------|-------------------|
| **ARIMA(2,1,2)** | **2988.04** | **3005.42** | **-1489.02** | **1st** |
| **ARIMA(1,1,3)** | 2988.38 | 3005.76 | -1489.19 | 2nd |
| **ARIMA(3,1,2)** | 2988.79 | 3009.65 | -1489.39 | 3rd |

**Selected Model: ARIMA(2,1,2)**
- **AR Coefficients**: φ1 = 1.3872, φ2 = -0.4754 (both significant)
- **MA Coefficients**: θ1 = -1.6519, θ2 = 0.7877 (both significant)
- **Differencing**: d = 1 (addresses non-stationarity)
- **Model Validation**: Ljung-Box test confirms white noise residuals

### **ARCH/GARCH Analysis for DJIA Returns**

**[INSERT ACF/PACF PLOTS FOR RETURNS AND SQUARED RETURNS HERE]**

**ARCH Effects Detection:**
- **Returns ACF**: Minimal significant autocorrelation (few lags significant)
- **Squared Returns ACF**: Strong autocorrelation through lag 20+
- **Ljung-Box Test**: p-values ≈ 10^-229 (highly significant ARCH effects)
- **Conclusion**: ARCH/GARCH modeling strongly indicated

**ARCH Model Selection Results:**
| **Model** | **AIC** | **BIC** | **Log Likelihood** | **Performance** |
|-----------|---------|---------|-------------------|----------------|
| **ARCH(1)** | 3712.75 | 3728.27 | -1854.38 | Poor |
| **ARCH(5)** | 3498.08 | 3534.29 | -1744.04 | Good |
| **ARCH(9)** | **3464.22** | **3521.11** | **-1721.11** | **Best** |

**GARCH Model Selection Results:**
| **Model** | **AIC** | **BIC** | **Log Likelihood** | **Significance** |
|-----------|---------|---------|-------------------|-----------------|
| **GARCH(1,1)** | **3449.12** | **3469.80** | **-1720.56** | **Optimal** |
| **GARCH(1,2)** | 3451.12 | 3476.97 | -1720.56 | Good |
| **GARCH(2,2)** | 3450.39 | 3481.42 | -1720.20 | Good |

## **Model Performance & Results**

### **ARIMA(2,1,2) Forecasting Performance**

**Train-Test Split Analysis:**
- **Training Period**: January 2000 - December 2015 (192 observations)
- **Test Period**: January 2016 - December 2019 (48 observations)
- **Forecast Horizon**: 48 months ahead

 ![image](https://github.com/user-attachments/assets/20821bf5-fdfd-4898-9274-657a7d8c3da8)


**Forecast Evaluation Metrics:**
- **MAPE**: 2.59% (excellent accuracy)
- **RMSE**: 290.54 (reasonable precision)
- **MAE**: 254.78 (consistent with RMSE)

**Temporal Error Patterns:**
- **Best Performance**: January (1.49% MAPE), 2017 (1.67% MAPE)
- **Worst Performance**: December (4.23% MAPE), 2019 (3.92% MAPE)
- **Error Trend**: Increasing forecast error over time (model degradation)

### **GARCH(1,1) Volatility Modeling**

**Selected Model Parameters:**
```
μ = 0.0540 (mean return)
ω = 0.0413 (unconditional variance)
α₁ = 0.1117 (ARCH term - volatility shocks)
β₁ = 0.8400 (GARCH term - volatility persistence)
```

**Model Interpretation:**
- **Persistence**: α₁ + β₁ = 0.9517 (high volatility persistence)
- **Half-life**: Approximately 13 days for volatility shocks to decay
- **Risk Premium**: Positive mean return compensating for volatility

**Model Comparison:**
| **Criterion** | **ARCH(9)** | **GARCH(1,1)** | **Advantage** |
|---------------|-------------|----------------|---------------|
| **AIC** | 3464.22 | **3449.12** | GARCH superior |
| **BIC** | 3521.11 | **3469.80** | GARCH superior |
| **Parameters** | 11 | 4 | GARCH more parsimonious |
| **Persistence** | Complex | Simple | GARCH interpretable |


![image](https://github.com/user-attachments/assets/d1f62bae-8fce-4039-8c43-a35ef8add1f8)


## **Diagnostic Testing & Model Validation**

### **ARIMA Model Diagnostics**
- **Residual Autocorrelation**: Ljung-Box p-values > 0.05 (white noise confirmed)
- **Normality**: Jarque-Bera test significant (non-normal residuals expected)
- **Heteroskedasticity**: Constant variance assumption satisfied
- **Model Adequacy**: Successfully captures temporal dependencies

### **GARCH Model Diagnostics**

![image](https://github.com/user-attachments/assets/8cb83fe9-bbaa-41f6-a78d-e5a1a44f19ec)


**Standardized Residuals Analysis:**
- **Time Plot**: No obvious patterns or clustering
- **QQ Plot**: Some deviation from normality in tails
- **ACF Analysis**: No significant autocorrelation in standardized residuals
- **Squared ACF**: Minimal autocorrelation in squared standardized residuals

**Model Validation Results:**
- **ARCH Effects Removal**: Successful elimination of heteroskedasticity
- **Forecast Accuracy**: Reliable volatility predictions for risk management
- **Economic Significance**: Captures market stress periods effectively

## **Implementation Guide**

### **Technical Requirements**
```python
# Core packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
```

### **ARIMA Implementation Workflow**

**Step 1: Data Preparation and Stationarity Testing**
```python
# Load and preprocess data
rsfhfs = pd.read_csv('RSFHFS.csv', index_col=0, parse_dates=True)
rsfhfs = rsfhfs['2000':'2019']

# Test stationarity
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(rsfhfs['RSFHFS'])
```

**Step 2: Model Selection and Estimation**
```python
# Systematic model selection
for p in range(4):
    for d in [1]:
        for q in range(4):
            model = ARIMA(rsfhfs['RSFHFS'], order=(p,d,q))
            results = model.fit()
            print(f"ARIMA({p},{d},{q}): AIC={results.aic:.2f}")
```

**Step 3: Forecasting and Evaluation**
```python
# Train-test split and forecasting
train_data = rsfhfs.loc[:'2015-12-31']
test_data = rsfhfs.loc['2016-01-01':]

model = ARIMA(train_data['RSFHFS'], order=(2,1,2))
model_fit = model.fit()
forecast = model_fit.get_forecast(steps=len(test_data))
```

### **ARCH/GARCH Implementation Workflow**

**Step 1: Return Calculation and Preliminary Analysis**
```python
# Calculate returns and check for ARCH effects
djia['ret'] = djia['DJIA'].pct_change() * 100
djia['sqret'] = djia['ret'] ** 2

# Test for ARCH effects
lb_test = acorr_ljungbox(djia['sqret'], lags=[10,20,30])
```

**Step 2: ARCH/GARCH Model Selection**
```python
# ARCH model selection
for p in range(1, 11):
    arch_model_p = arch_model(djia['ret'], vol='ARCH', p=p)
    arch_result = arch_model_p.fit(disp='off')
    print(f"ARCH({p}): AIC={arch_result.aic:.2f}")

# GARCH model selection  
garch_model = arch_model(djia['ret'], vol='GARCH', p=1, q=1)
garch_result = garch_model.fit(disp='off')
```

**Step 3: Volatility Forecasting**
```python
# Generate volatility forecasts
forecast = garch_result.forecast(horizon=10)
forecasted_variance = forecast.variance.iloc[-1].values
volatility_forecast = np.sqrt(forecasted_variance) * 100
```

## **Business Applications & Investment Implications**

### **Retail Sales Forecasting Applications**
- **Inventory Management**: Optimize stock levels based on sales predictions
- **Strategic Planning**: Long-term capacity and expansion decisions
- **Market Timing**: Identify optimal periods for product launches
- **Economic Indicators**: Early warning signals for economic downturns

### **Volatility Modeling Applications**
- **Risk Management**: VaR calculations and stress testing
- **Portfolio Optimization**: Dynamic hedging and asset allocation
- **Derivative Pricing**: Option valuation and volatility trading
- **Regulatory Compliance**: Capital adequacy and risk reporting

### **Strategic Recommendations**

#### **For Retail Analysts**
- **Seasonal Adjustment**: Account for December underperformance in forecast models
- **Economic Integration**: Incorporate macroeconomic indicators for improved accuracy
- **Model Updating**: Implement rolling window estimation for parameter stability

#### **For Risk Managers**
- **Volatility Clustering**: Prepare for extended high-volatility periods
- **Persistence Monitoring**: Track α + β sum for regime change detection
- **Stress Testing**: Use GARCH forecasts for extreme scenario planning

### **Model Limitations & Future Enhancements**

**Current Limitations:**
- **ARIMA**: Increasing forecast error over extended horizons
- **GARCH**: Assumes normal distribution (fat tails not fully captured)
- **Parameter Stability**: Models may require periodic re-estimation

**Recommended Extensions:**
- **Regime-Switching Models**: Capture structural breaks and crisis periods
- **Multivariate GARCH**: Model cross-asset volatility spillovers
- **Machine Learning Integration**: Neural networks for non-linear volatility patterns
- **High-Frequency Models**: Realized volatility and jump detection

---

*This analysis demonstrates comprehensive application of ARCH/GARCH modeling techniques to real-world financial and economic data, providing robust frameworks for volatility forecasting and risk management in institutional settings.*
