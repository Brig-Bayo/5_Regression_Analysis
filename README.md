# Module 5: Linear and Non-linear Regression

## Introduction to Regression Analysis

Regression analysis is a statistical method used to examine the relationship between a dependent variable (outcome) and one or more independent variables (predictors). It helps us understand how the dependent variable changes when the independent variables are varied, allowing us to make predictions and inferences about the relationship.

In this module, we'll explore:
1. Simple linear regression
2. Multiple linear regression
3. Polynomial regression
4. Generalized Additive Models (GAMs)
5. Logistic regression

## Required Packages for This Module

```r
# Install required packages (only need to do this once)
install.packages(c("ggplot2", "dplyr", "car", "lmtest", "MASS", "mgcv", "performance", "see"))

# Load the packages
library(ggplot2)     # For data visualization
library(dplyr)       # For data manipulation
library(car)         # For regression diagnostics
library(lmtest)      # For testing regression assumptions
library(MASS)        # For additional regression functions
library(mgcv)        # For GAMs
library(performance) # For model performance metrics
library(see)         # For visualization of model diagnostics
```

## Simple Linear Regression

Simple linear regression models the relationship between two continuous variables: a dependent variable (Y) and a single independent variable (X). The relationship is modeled as a straight line:

Y = β₀ + β₁X + ε

Where:
- Y is the dependent variable
- X is the independent variable
- β₀ is the intercept (value of Y when X = 0)
- β₁ is the slope (change in Y for a one-unit change in X)
- ε is the error term (residual)

### Assumptions of Linear Regression

1. **Linearity**: The relationship between X and Y is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: The variance of residuals is constant across all values of X
4. **Normality**: The residuals are normally distributed
5. **No multicollinearity**: Independent variables are not highly correlated (relevant for multiple regression)

### Interpretation of Simple Linear Regression Results

When interpreting simple linear regression results, consider:

1. **R-squared**: The proportion of variance in Y explained by X (ranges from 0 to 1)
2. **Coefficient significance**: p-values for the intercept and slope
3. **Coefficient values**: The intercept (β₀) and slope (β₁)
4. **Residual standard error**: The average deviation of observations from the regression line
5. **F-statistic**: Tests if the model as a whole is significant

## Multiple Linear Regression

Multiple linear regression extends simple linear regression to include multiple independent variables:

Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε

Where:
- Y is the dependent variable
- X₁, X₂, ..., Xₚ are the independent variables
- β₀, β₁, β₂, ..., βₚ are the coefficients
- ε is the error term


### Model Selection in Multiple Regression

When building multiple regression models, it's important to select the most appropriate set of predictors. Common approaches include:

1. **Stepwise Selection**: Adding or removing predictors based on statistical criteria
2. **Information Criteria**: Using AIC or BIC to compare models
3. **Cross-Validation**: Evaluating model performance on held-out data

## Polynomial Regression

Polynomial regression extends linear regression by including polynomial terms of the independent variables:

Y = β₀ + β₁X + β₂X² + ... + βₙXⁿ + ε

This allows modeling of non-linear relationships while still using the linear regression framework.

## Generalized Additive Models (GAMs)

Generalized Additive Models (GAMs) extend linear models by allowing non-linear relationships between the dependent variable and independent variables. Instead of specifying a particular form for the non-linearity (like in polynomial regression), GAMs use smooth functions:

Y = β₀ + f₁(X₁) + f₂(X₂) + ... + fₚ(Xₚ) + ε

Where f₁, f₂, ..., fₚ are smooth functions (often splines) of the predictors.


## Logistic Regression

Logistic regression is used when the dependent variable is binary (0/1, Yes/No, Success/Failure). It models the probability of the dependent variable being 1 given the independent variables:

log(p/(1-p)) = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ

Where p is the probability of the dependent variable being 1.

