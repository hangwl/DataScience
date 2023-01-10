# Regression Analysis

In regression analysis, we draw a random sample from a population and use it to estimate the properties of that population.
In our regression equation, the coefficients are estimates of the actual population parameters, which we would like to be the best such that they are unbiased and minimize discrepancies between data.

In econometric analysis, OLS estimators are typical. Assuming Gauss-Markov Assumptions hold, OLS estimates are Best Linear Unbiased Estimators (BLUE).
i.e. they are unbiased and efficient

# Gauss-Markov Assumptions:
1. Linearity in parameters
2. Random Sampling
3. No perfect collinearity among covariates
    If an independent variable is an exact linear combination of other independent variables, we say that the model suffers from perfect collinearity, and it cannot be estimated by OLS.
4. Zero conditional Mean (Exogeneity)
    In general, we should expect error terms to be distributed around the mean of zero. If a key variable has been omitted, it can cause omitted variable bias since the omitted variable could be correlated with independent variables.
5. Homoskedasticity (constant variance) of errors
    Heteroskedasticity causes our coefficient estimates to be less precise. To test for homoskedasticity, we can use the Breusch-Pagan test, where the alternative hypothesis is that the homoskedasticity assumption does not hold. We can also opt to use robust standard errors in the presence of heteroskedasticity (given a sufficiently large sample).

# Data Structures
1. Cross-Sectional Data
2. Time Series Data
3. Panel Data