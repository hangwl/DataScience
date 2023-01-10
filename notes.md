# Regression Analysis

In regression analysis, we draw a random sample from a population and use it to estimate the properties of that population. In our regression equation, the coefficients are estimates of the actual population parameters, which we would like to be the best such that they are unbiased and minimize discrepancies between data.

In econometric analysis, OLS estimators are typical. Assuming Gauss-Markov Assumptions hold, OLS estimates are Best Linear Unbiased Estimators (BLUE), i.e. they are unbiased and efficient.

# Gauss-Markov Assumptions:
1. Linearity in Parameters
2. Random Sampling
3. No Perfect Collinearity among Covariates
   
   If an independent variable is an exact linear combination of other independent variables, we say that the model suffers from perfect collinearity, and it cannot be estimated by OLS.

4. Zero Conditional Mean (Exogeneity)
   
   In general, we should expect error terms to be distributed around the mean of zero. If a key variable has been omitted, it can cause omitted variable bias since the omitted variable could be correlated with independent variables.

5. Homoskedasticity (constant variance) of errors
   
   Heteroskedasticity causes our coefficient estimates to be less precise. To test for homoskedasticity, we can use the Breusch-Pagan test, where the alternative hypothesis is that the homoskedasticity assumption does not hold. We can also opt to use robust standard errors in the presence of heteroskedasticity (given a sufficiently large sample).

# Time Series Analysis

In applied time series analysis, we are concerned with the dynamic consequences of events over time. 

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

### Stationarity

In practice, we should always test the stationarity of time series data before running a regression model. If the time series data is not stationary, then we cannot run the regression model by using the raw data directly.

### Static Model

In a static model, a change in X immediately affects y.

### Finite Distributed Lags (FDL) Model

In a FDL model, we allow one or more variables to affect y with a lag. We specify our model to include n impact multipliers for an FDL of order n. The long run propensity is measured as the sum of the estimated coefficients associated with x and its lags.