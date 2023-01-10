
- [Regression Analysis](#regression-analysis)
  - [Gauss-Markov Assumptions:](#gauss-markov-assumptions)
- [Time Series Analysis](#time-series-analysis)
  - [Stationarity](#stationarity)
  - [Static Model](#static-model)
  - [Finite Distributed Lags (FDL) Model](#finite-distributed-lags-fdl-model)
  - [AutoRegressive Integrated Moving Average (ARIMA) Model](#autoregressive-integrated-moving-average-arima-model)
    - [AutoRegressive (AR) Process](#autoregressive-ar-process)
    - [MovingAverage (MA) Process](#movingaverage-ma-process)
  - [Simple Forecast](#simple-forecast)
  - [Forecast Evaluation](#forecast-evaluation)
  - [Box-Jenkins Approach](#box-jenkins-approach)

# Regression Analysis

In regression analysis, we draw a random sample from a population and use it to estimate the properties of that population. In our regression equation, the coefficients are estimates of the actual population parameters, which we would like to be the best such that they are unbiased and minimize discrepancies between data.

In econometric analysis, OLS estimators are typical. Assuming Gauss-Markov Assumptions hold, OLS estimates are Best Linear Unbiased Estimators (BLUE), i.e. they are unbiased and efficient.

## Gauss-Markov Assumptions:
1. Linearity in Parameters
2. Random Sampling
3. No Perfect Collinearity among Covariates
   
   If an independent variable is an exact linear combination of other independent variables, we say that the model suffers from perfect collinearity, and it cannot be estimated by OLS.

4. Zero Conditional Mean (Exogeneity)
   
   In general, we should expect error terms to be distributed around the mean of zero. If a key variable has been omitted, it can cause omitted variable bias since the omitted variable could be correlated with independent variables.

5. Homoskedasticity (constant variance) of errors
   
   Heteroskedasticity causes our coefficient estimates to be less precise. To test for homoskedasticity, we can use the Breusch-Pagan test, where the alternative hypothesis is that the homoskedasticity assumption does not hold. We can also opt to use robust standard errors in the presence of heteroskedasticity (given a sufficiently large sample).

## Addressing Endogeneities

Possible Sources:
- Omitted Variable Bias
- Measurement Error
- Selection Bias
- Misspecification of Functional Form
- Reverse Causality




# Time Series Analysis

In applied time series analysis, we are concerned with the dynamic consequences of events over time. We rely on the Box-Jenkins approach to generate time series forecasts.

OLS can also be used to estimate time series data under similar assumptions:
1. Linearity in Parameters
2. Zero Conditional Mean
3. No Perfect Collinearity
4. Homoskedasticity
   * To test for homoskedasticity, we can use the Breusch-Pagan test.
   * To remedy heteroskedasticity, we can use the robust standard procedure estimation.
   * To model conditional heteroskedasticity, we can rely on the autoregressive conditional heteroskedasticity (ARCH) model.
5. No Serial Correlation
   * Conditional on X, the errors in two different time periods should be uncorrelated.
   * We can use the Breusch-Godfrey (BG) Test as a general test for autocorrelation vs Durbin Watson (DW) test.
   * The BG test generalizes to any order autocorrelation, and allows the original regression model to contain lagged dependent variables.
   * To fix autocorrelation, we can use feasible generalized least squares (FGLS) instead of OLS, or include lagged dependent variables and lagged X. We should note that FGLS requires strictly exogenous explanatory variables, and should not be used when the explanatory variables include lagged dependent variables.

Under TS assumptions 1-5, OLS estimators are BLUE condiional on X.

1. Normality of Errors

For time series data, the assumptions for strict exogeneity and no serial correlation are often unrealistic. We should note that if the homoskedasticity and/or autocorrelation assumptions are violated, the usual OLS estimators no longer minimize variance among all linear unbiased estimators. As a result, usual t and F tests become invalid.

## Stationarity

In practice, we should always test the stationarity of time series data before running a regression model. If the time series data is not stationary, then we cannot run the regression model by using the raw data directly as it will have a time-varying mean or time-varying variance or both. For example, exchange rates and housing prices are usually non-stationary.

If a time series is not stationary, we can study its behavior only for the time period under consideration, and forecasts will have little practical value.

## Static Model

In a static model, a change in X immediately affects y.

## Finite Distributed Lags (FDL) Model

In a FDL model, we allow one or more variables to affect y with a lag. We specify our model to include n impact multipliers for an FDL of order n. The long run propensity is measured as the sum of the estimated coefficients associated with x and its lags.

## AutoRegressive Integrated Moving Average (ARIMA) Model

ARIMA is useful when a structural model is inappropriate or unknown, where a structural model is trying to explain the changes in a variable by reference to the movements in the current and past values of other explanatory variables.

An ARMA model is a stationary model. If our model isn't stationary, we can achieve stationarity by taking a series of differenecs. The 'I' in ARIMA model stands for integrated i.e. it is a measure of how many non-seasonal differences are needed to achieve stationarity.

We can use the graphs of ACF and PACF (autocorrelation and partial autocorrelation functions) to help us decide on the order of (p) and (q) for our AR(p) and MA(q) models.\
<https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf>\
<https://people.duke.edu/~rnau/411arim3.htm>

General Guideline:

| Model     | ACF Patterns                                                      | PACF Patterns                                                     |
|-----------|-------------------------------------------------------------------|-------------------------------------------------------------------|
| AR(p)     | Decays exponentially and/or displays damped sine wave pattern     | Displays significant spikes through p lags (cuts off after lag p) |
| MA(q)     | Displays significant spikes through q lags (cuts off after lag q) | Decays exponentially                                              |
| ARMA(p,q) | Decays exponentially                                              | Decays exponentially                                              |

Akaike Information Criterion (AIC) and Schwarz Bayesian Criterion (BIC) are more sophisticated model-selection statistics widely used to determine if the ARMA(p,q) or AR(p) or MA(q) model is a good statistical fit. A smaller AIC (or BIC) indicates a better fitting model.

We should note that although they usually give the same conclusion, BIC will tend to always select a more parsimonious model than the AIC.

### AutoRegressive (AR) Process

In an AR(p) model, the dependent variable at time t depends on its value in the previous period(s) and a random error term.

### MovingAverage (MA) Process

The moving average model is simply a linear combination of random error terms. Therefore, our dependent variable depends on the current and previous values of the random error terms, which are white noise.

## Simple Forecast

Suppose we know the true generating process, we can obtain a simple forecast for period t+1 via its respective forecast function. In j-step ahead forecasts, given that the forecast function yield a convergent sequence of forecasts, the conditional forecast of y_{t+j} converges to unconditional expectation.

## Forecast Evaluation

To evaluate our forecasts, we can apply a iterative scheme to use 90% of our observations to estimate the competing models. That is, we can easily calculated the 1-step ahead forecast errors for e_{t=91} since we already know the realization of y_{t=91}. We then do the same for t=92, t=93, ..., and obtain 10 forecast errors for our model(s).

The evaluation criterion depends is subjective. Some examples are:
* Mean Squared Error (MSE)
* Mean Error (ME)
* Mean Percentage Error (MPE)
* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)

## Box-Jenkins Approach

The general time series forecast procedure is as follows:

1. Identification
   * consider stationarity
     * if non-stationary, consider taking the (first) difference on our y variable
       * see unit root tests
     * if stationary, consider correlograms to decide models and order of lags
2. Estimation
   * if model cannot be decided in stage 1, use AIC (or BIC) criterion
   * run regressions
3. Diagnostic Checking
   * we need to include AR and MA terms to ensure that the residual terms are effectively white noise
     * the coefficients of the p and q lags must be significant, but the interim lags need not be (skip them if they are not useful)
     * Test the residuals
4. Forecasting (either 1-step ahead or j-step ahead)