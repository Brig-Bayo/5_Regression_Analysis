# Module 5: Linear and Non-linear Regression

This repository provides R scripts and examples for performing various regression analyses, including linear, polynomial, generalized additive, and logistic regression. These models are fundamental tools for understanding and predicting relationships between variables in data science and statistics.

---

## Table of Contents

- [Introduction to Regression Analysis](#introduction-to-regression-analysis)
- [Required Packages](#required-packages)
- [Simple Linear Regression](#simple-linear-regression)
  - [Assumptions of Linear Regression](#assumptions-of-linear-regression)
  - [Interpretation of Simple Linear Regression](#interpretation-of-simple-linear-regression)
- [Multiple Linear Regression](#multiple-linear-regression)
  - [Model Selection in Multiple Regression](#model-selection-in-multiple-regression)
- [Polynomial Regression](#polynomial-regression)
- [Generalized Additive Models (GAMs)](#generalized-additive-models-gams)
- [Logistic Regression](#logistic-regression)
- [References](#references)

---

## Introduction to Regression Analysis

Regression analysis is a statistical method used to examine the relationship between a dependent variable (outcome) and one or more independent variables (predictors). It helps us understand how the dependent variable changes as the independent variables vary, allowing for both prediction and inference.

In this module, we cover:
1. Simple linear regression
2. Multiple linear regression
3. Polynomial regression
4. Generalized Additive Models (GAMs)
5. Logistic regression

---

## Required Packages

Make sure you install and load the following R packages:

```r
# Install required packages (run once)
install.packages(c("ggplot2", "dplyr", "car", "lmtest", "MASS", "mgcv", "performance", "see"))

# Load the packages
library(ggplot2)     # Data visualization
library(dplyr)       # Data manipulation
library(car)         # Regression diagnostics
library(lmtest)      # Testing regression assumptions
library(MASS)        # Additional regression functions
library(mgcv)        # Generalized Additive Models
library(performance) # Model performance metrics
library(see)         # Visualization of model diagnostics
```
---
## Simple Linear Regression

Simple linear regression models the relationship between a continuous dependent variable (Y) and a single independent variable (X):

```r
Y = β₀ + β₁X + ε
```
Where:

Y: Dependent variable
X: Independent variable
β₀: Intercept (value of Y when X = 0)
β₁: Slope (change in Y for a one-unit change in X)
ε: Error term (residual)

## Assumptions of Linear Regression

Linearity: The relationship between X and Y is linear.

Independence: Observations are independent of each other.

Homoscedasticity: The variance of residuals is constant across all values of X.

Normality: The residuals are normally distributed.

No multicollinearity: (For multiple regression) Independent variables are not highly correlated.

## Interpretation of Simple Linear Regression

R-squared: Proportion of variance in Y explained by X (0 to 1).

Coefficient significance: p-values for intercept and slope.

Coefficient values: The intercept (β₀) and slope (β₁).

Residual standard error: Average deviation of observations from the regression line.

F-statistic: Tests overall model significance.

---
## Multiple Linear Regression

Multiple linear regression extends simple linear regression by including multiple independent variables:
```r
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
```
Where:

Y: Dependent variable

X₁, X₂, ..., Xₚ: Independent variables

β₀, β₁, ..., βₚ: Coefficients

ε: Error term

## Model Selection in Multiple Regression

Selecting the best set of predictors is crucial. Common approaches include:

Stepwise Selection: Add or remove predictors based on statistical criteria.

Information Criteria: Compare models using AIC or BIC.

Cross-Validation: Evaluate model performance on held-out data.

---
## Polynomial Regression

Polynomial regression includes polynomial terms of the independent variables:

```r
Y = β₀ + β₁X + β₂X² + ... + βₙXⁿ + ε
```
This allows modeling of non-linear relationships within the linear regression framework.

---

## Generalized Additive Models (GAMs)

Generalized Additive Models (GAMs) extend linear models by allowing non-linear relationships between the dependent variable and independent variables using smooth functions (splines):

```r
Y = β₀ + f₁(X₁) + f₂(X₂) + ... + fₚ(Xₚ) + ε
```
Where f₁, f₂, ..., fₚ are smooth functions of the predictors.

---

## Logistic Regression

```r
log(p/(1-p)) = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ
```
Where ***p*** is the probability of the outcome being 1.

---
## References

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning.

Fox, J., & Weisberg, S. (2018). An R Companion to Applied Regression.

R Documentation: lm(), glm(), mgcv

---


