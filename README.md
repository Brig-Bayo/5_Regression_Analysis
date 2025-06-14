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

### Example 1: House Prices and Size

Let's examine the relationship between house size (in square feet) and house price.

```r
# Create a dataset of house sizes and prices
set.seed(123)  # For reproducibility
house_data <- data.frame(
  size_sqft = seq(1000, 3000, by = 100),
  price_thousands = 100 + 0.2 * seq(1000, 3000, by = 100) + rnorm(21, 0, 15)
)

# View the first few rows
head(house_data)

# Create a scatter plot
ggplot(house_data, aes(x = size_sqft, y = price_thousands)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  labs(title = "House Price vs. Size",
       x = "House Size (square feet)",
       y = "House Price (thousands $)") +
  theme_minimal()
```

#### Step 1: Fit a Simple Linear Regression Model

```r
# Fit a simple linear regression model
house_model <- lm(price_thousands ~ size_sqft, data = house_data)

# View the model summary
summary(house_model)
```

#### Step 2: Interpret the Coefficients

```r
# Extract coefficients
coef(house_model)

# Confidence intervals for coefficients
confint(house_model, level = 0.95)
```

#### Step 3: Check Model Assumptions

```r
# Create diagnostic plots
par(mfrow = c(2, 2))
plot(house_model)
par(mfrow = c(1, 1))

# Alternative: Use performance package for nicer diagnostics
check_model(house_model)

# Test for normality of residuals
shapiro.test(residuals(house_model))

# Test for homoscedasticity
bptest(house_model)  # Breusch-Pagan test
```

#### Step 4: Make Predictions

```r
# Create new data for prediction
new_houses <- data.frame(size_sqft = c(1500, 2200, 2800))

# Make predictions
predictions <- predict(house_model, newdata = new_houses, interval = "prediction")
cbind(new_houses, predictions)

# Visualize the regression line with confidence and prediction intervals
ggplot(house_data, aes(x = size_sqft, y = price_thousands)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", formula = y ~ x, color = "red", fill = "pink") +
  labs(title = "Simple Linear Regression: House Price vs. Size",
       x = "House Size (square feet)",
       y = "House Price (thousands $)") +
  theme_minimal()
```

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

### Example 2: House Prices with Multiple Predictors

Let's extend our house price example to include multiple predictors: size, number of bedrooms, and age of the house.

```r
# Create a dataset with multiple predictors
set.seed(123)  # For reproducibility
house_data_multi <- data.frame(
  size_sqft = seq(1000, 3000, by = 100),
  bedrooms = sample(2:5, 21, replace = TRUE),
  age_years = sample(1:30, 21, replace = TRUE),
  price_thousands = numeric(21)
)

# Generate prices based on a formula with some random noise
for (i in 1:nrow(house_data_multi)) {
  house_data_multi$price_thousands[i] <- 50 + 
    0.15 * house_data_multi$size_sqft[i] + 
    15 * house_data_multi$bedrooms[i] - 
    2 * house_data_multi$age_years[i] + 
    rnorm(1, 0, 20)
}

# View the first few rows
head(house_data_multi)
```

#### Step 1: Explore the Data

```r
# Create scatter plots for each predictor
par(mfrow = c(1, 3))
plot(house_data_multi$size_sqft, house_data_multi$price_thousands, 
     main = "Price vs. Size", xlab = "Size (sqft)", ylab = "Price ($K)")
plot(house_data_multi$bedrooms, house_data_multi$price_thousands, 
     main = "Price vs. Bedrooms", xlab = "Bedrooms", ylab = "Price ($K)")
plot(house_data_multi$age_years, house_data_multi$price_thousands, 
     main = "Price vs. Age", xlab = "Age (years)", ylab = "Price ($K)")
par(mfrow = c(1, 1))

# Check correlations between variables
cor(house_data_multi)
```

#### Step 2: Fit a Multiple Linear Regression Model

```r
# Fit a multiple linear regression model
house_model_multi <- lm(price_thousands ~ size_sqft + bedrooms + age_years, 
                        data = house_data_multi)

# View the model summary
summary(house_model_multi)
```

#### Step 3: Check for Multicollinearity

```r
# Calculate Variance Inflation Factors (VIF)
vif(house_model_multi)
```

#### Step 4: Check Model Assumptions

```r
# Create diagnostic plots
par(mfrow = c(2, 2))
plot(house_model_multi)
par(mfrow = c(1, 1))

# Alternative: Use performance package for nicer diagnostics
check_model(house_model_multi)
```

#### Step 5: Make Predictions

```r
# Create new data for prediction
new_houses_multi <- data.frame(
  size_sqft = c(1500, 2200, 2800),
  bedrooms = c(3, 4, 5),
  age_years = c(10, 5, 15)
)

# Make predictions
predictions_multi <- predict(house_model_multi, newdata = new_houses_multi, 
                            interval = "prediction")
cbind(new_houses_multi, predictions_multi)
```

### Model Selection in Multiple Regression

When building multiple regression models, it's important to select the most appropriate set of predictors. Common approaches include:

1. **Stepwise Selection**: Adding or removing predictors based on statistical criteria
2. **Information Criteria**: Using AIC or BIC to compare models
3. **Cross-Validation**: Evaluating model performance on held-out data

#### Example: Stepwise Selection

```r
# Fit a full model
full_model <- lm(price_thousands ~ size_sqft + bedrooms + age_years, 
                data = house_data_multi)

# Perform stepwise selection
step_model <- step(full_model, direction = "both", trace = TRUE)

# View the final model
summary(step_model)
```

#### Example: Comparing Models with AIC

```r
# Fit different models
model1 <- lm(price_thousands ~ size_sqft, data = house_data_multi)
model2 <- lm(price_thousands ~ size_sqft + bedrooms, data = house_data_multi)
model3 <- lm(price_thousands ~ size_sqft + bedrooms + age_years, data = house_data_multi)

# Compare AIC values
AIC(model1, model2, model3)
```

## Polynomial Regression

Polynomial regression extends linear regression by including polynomial terms of the independent variables:

Y = β₀ + β₁X + β₂X² + ... + βₙXⁿ + ε

This allows modeling of non-linear relationships while still using the linear regression framework.

### Example 3: Non-linear Relationship

Let's create a dataset with a non-linear relationship between X and Y.

```r
# Create a dataset with a non-linear relationship
set.seed(123)  # For reproducibility
nonlinear_data <- data.frame(
  x = seq(-3, 3, by = 0.2),
  y = numeric(31)
)

# Generate y values based on a quadratic function with some random noise
for (i in 1:nrow(nonlinear_data)) {
  nonlinear_data$y[i] <- 2 + 1.5 * nonlinear_data$x[i] - 2 * nonlinear_data$x[i]^2 + rnorm(1, 0, 1)
}

# View the first few rows
head(nonlinear_data)

# Create a scatter plot
ggplot(nonlinear_data, aes(x = x, y = y)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  labs(title = "Non-linear Relationship",
       x = "X",
       y = "Y") +
  theme_minimal()
```

#### Step 1: Fit a Linear Model (for comparison)

```r
# Fit a simple linear model
linear_model <- lm(y ~ x, data = nonlinear_data)

# View the model summary
summary(linear_model)
```

#### Step 2: Fit a Polynomial Model

```r
# Fit a quadratic model (polynomial of degree 2)
quadratic_model <- lm(y ~ x + I(x^2), data = nonlinear_data)

# View the model summary
summary(quadratic_model)

# Fit a cubic model (polynomial of degree 3)
cubic_model <- lm(y ~ x + I(x^2) + I(x^3), data = nonlinear_data)

# View the model summary
summary(cubic_model)
```

#### Step 3: Compare Models

```r
# Compare models using ANOVA
anova(linear_model, quadratic_model, cubic_model)

# Compare models using AIC
AIC(linear_model, quadratic_model, cubic_model)
```

#### Step 4: Visualize the Models

```r
# Create a grid of x values for prediction
grid <- data.frame(x = seq(min(nonlinear_data$x), max(nonlinear_data$x), length.out = 100))

# Make predictions for each model
grid$linear <- predict(linear_model, newdata = grid)
grid$quadratic <- predict(quadratic_model, newdata = grid)
grid$cubic <- predict(cubic_model, newdata = grid)

# Plot the data and model predictions
ggplot(nonlinear_data, aes(x = x, y = y)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  geom_line(data = grid, aes(y = linear, color = "Linear"), size = 1) +
  geom_line(data = grid, aes(y = quadratic, color = "Quadratic"), size = 1) +
  geom_line(data = grid, aes(y = cubic, color = "Cubic"), size = 1) +
  scale_color_manual(name = "Model", 
                     values = c("Linear" = "red", "Quadratic" = "green", "Cubic" = "purple")) +
  labs(title = "Comparison of Linear and Polynomial Models",
       x = "X",
       y = "Y") +
  theme_minimal()
```

### Alternative Approach: Using the poly() Function

R provides the `poly()` function for creating orthogonal polynomials, which can help with numerical stability:

```r
# Fit polynomial models using poly()
poly_model2 <- lm(y ~ poly(x, 2), data = nonlinear_data)  # Degree 2
poly_model3 <- lm(y ~ poly(x, 3), data = nonlinear_data)  # Degree 3

# View the model summaries
summary(poly_model2)
summary(poly_model3)
```

## Generalized Additive Models (GAMs)

Generalized Additive Models (GAMs) extend linear models by allowing non-linear relationships between the dependent variable and independent variables. Instead of specifying a particular form for the non-linearity (like in polynomial regression), GAMs use smooth functions:

Y = β₀ + f₁(X₁) + f₂(X₂) + ... + fₚ(Xₚ) + ε

Where f₁, f₂, ..., fₚ are smooth functions (often splines) of the predictors.

### Example 4: GAM for Non-linear Relationships

Let's use the same non-linear dataset and fit a GAM.

```r
# Fit a GAM
gam_model <- gam(y ~ s(x), data = nonlinear_data)

# View the model summary
summary(gam_model)
```

#### Step 1: Visualize the GAM

```r
# Plot the GAM
plot(gam_model, residuals = TRUE, pch = 19, cex = 1)

# Create a grid of x values for prediction
grid <- data.frame(x = seq(min(nonlinear_data$x), max(nonlinear_data$x), length.out = 100))

# Make predictions for the GAM
grid$gam <- predict(gam_model, newdata = grid)

# Plot the data and GAM prediction
ggplot(nonlinear_data, aes(x = x, y = y)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  geom_line(data = grid, aes(y = gam), color = "red", size = 1) +
  labs(title = "Generalized Additive Model (GAM)",
       x = "X",
       y = "Y") +
  theme_minimal()
```

#### Step 2: Compare GAM with Polynomial Models

```r
# Make predictions for each model
grid$quadratic <- predict(quadratic_model, newdata = grid)

# Plot the data and model predictions
ggplot(nonlinear_data, aes(x = x, y = y)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  geom_line(data = grid, aes(y = quadratic, color = "Polynomial"), size = 1) +
  geom_line(data = grid, aes(y = gam, color = "GAM"), size = 1) +
  scale_color_manual(name = "Model", 
                     values = c("Polynomial" = "green", "GAM" = "red")) +
  labs(title = "Comparison of Polynomial and GAM Models",
       x = "X",
       y = "Y") +
  theme_minimal()
```

#### Step 3: Check GAM Assumptions

```r
# Check GAM residuals
gam.check(gam_model)
```

### GAMs with Multiple Predictors

GAMs can handle multiple predictors, allowing different smoothing functions for each:

```r
# Create a dataset with multiple predictors
set.seed(123)  # For reproducibility
multi_data <- data.frame(
  x1 = runif(100, -3, 3),
  x2 = runif(100, 0, 10),
  y = numeric(100)
)

# Generate y values based on non-linear functions of x1 and x2
for (i in 1:nrow(multi_data)) {
  multi_data$y[i] <- 2 - 2 * multi_data$x1[i]^2 + 0.5 * sin(multi_data$x2[i]) + rnorm(1, 0, 1)
}

# Fit a GAM with multiple predictors
multi_gam <- gam(y ~ s(x1) + s(x2), data = multi_data)

# View the model summary
summary(multi_gam)

# Plot the smooth terms
par(mfrow = c(1, 2))
plot(multi_gam, residuals = TRUE, pch = 19, cex = 0.7)
par(mfrow = c(1, 1))
```

## Logistic Regression

Logistic regression is used when the dependent variable is binary (0/1, Yes/No, Success/Failure). It models the probability of the dependent variable being 1 given the independent variables:

log(p/(1-p)) = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ

Where p is the probability of the dependent variable being 1.

### Example 5: Predicting Exam Pass/Fail

Let's create a dataset to predict whether a student will pass an exam based on study hours and previous GPA.

```r
# Create a dataset for exam pass/fail prediction
set.seed(123)  # For reproducibility
exam_data <- data.frame(
  study_hours = runif(100, 0, 10),
  prev_gpa = runif(100, 1.5, 4.0),
  pass = numeric(100)
)

# Generate pass/fail outcomes based on a logistic model
for (i in 1:nrow(exam_data)) {
  # Calculate the log-odds
  log_odds <- -5 + 0.8 * exam_data$study_hours[i] + 1.2 * exam_data$prev_gpa[i]
  
  # Convert log-odds to probability
  prob <- exp(log_odds) / (1 + exp(log_odds))
  
  # Generate binary outcome based on probability
  exam_data$pass[i] <- rbinom(1, 1, prob)
}

# Convert pass to a factor
exam_data$pass <- factor(exam_data$pass, levels = c(0, 1), labels = c("Fail", "Pass"))

# View the first few rows
head(exam_data)
```

#### Step 1: Explore the Data

```r
# Create scatter plots
ggplot(exam_data, aes(x = study_hours, y = prev_gpa, color = pass)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = c("Fail" = "red", "Pass" = "green"
(Content truncated due to size limit. Use line ranges to read in chunks)
