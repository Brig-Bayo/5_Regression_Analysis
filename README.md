# Module 5: Linear and Non-linear Regression

This repository provides R scripts and examples for performing a variety of regression analyses, including linear, polynomial, generalized additive, and logistic regression. These methods help explore and quantify relationships between variables, make predictions, and draw statistical inferences.

---

## Table of Contents

- [Introduction to Regression Analysis](#introduction-to-regression-analysis)
- [Required Packages](#required-packages)
- [Simple Linear Regression](#simple-linear-regression)
- [Assumptions of Linear Regression](#assumptions-of-linear-regression)
- [Multiple Linear Regression](#multiple-linear-regression)
- [Model Selection in Multiple Regression](#model-selection-in-multiple-regression)
- [Polynomial Regression](#polynomial-regression)
- [Generalized Additive Models (GAMs)](#generalized-additive-models-gams)
- [Logistic Regression](#logistic-regression)
- [References](#references)

---

## Introduction to Regression Analysis

Regression analysis is a statistical method used to examine the relationship between a dependent variable (outcome) and one or more independent variables (predictors). It helps us understand how the dependent variable changes as the independent variables vary, allowing for prediction and inference.

In this module, we cover:
1. Simple linear regression
2. Multiple linear regression
3. Polynomial regression
4. Generalized Additive Models (GAMs)
5. Logistic regression

---

## Required Packages

To run the analyses in this repository, install and load the following R packages:

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
