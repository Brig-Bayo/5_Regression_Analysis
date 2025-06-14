# Module 5: Linear and Non-linear Regression
# This script demonstrates linear and non-linear regression techniques

# ===== Load Required Packages =====
# Install required packages if not already installed
required_packages <- c("ggplot2", "dplyr", "car", "lmtest", "MASS", "mgcv", "performance", "see", "caret")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Load the packages
library(ggplot2)     # For data visualization
library(dplyr)       # For data manipulation
library(car)         # For regression diagnostics
library(lmtest)      # For testing regression assumptions
library(MASS)        # For additional regression functions
library(mgcv)        # For GAMs
library(performance) # For model performance metrics
library(see)         # For visualization of model diagnostics
library(caret)       # For model validation

# ===== Example 1: Simple Linear Regression (House Prices and Size) =====
# Create a dataset of house sizes and prices
set.seed(123)  # For reproducibility
house_data <- data.frame(
  size_sqft = seq(1000, 3000, by = 100),
  price_thousands = 100 + 0.2 * seq(1000, 3000, by = 100) + rnorm(21, 0, 15)
)

# View the first few rows
head(house_data)

# Create a scatter plot
house_scatter <- ggplot(house_data, aes(x = size_sqft, y = price_thousands)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  labs(title = "House Price vs. Size",
       x = "House Size (square feet)",
       y = "House Price (thousands $)") +
  theme_minimal()
print(house_scatter)

# Step 1: Fit a Simple Linear Regression Model
# Fit a simple linear regression model
house_model <- lm(price_thousands ~ size_sqft, data = house_data)

# View the model summary
summary(house_model)

# Step 2: Interpret the Coefficients
# Extract coefficients
coef_house <- coef(house_model)
print(coef_house)

# Confidence intervals for coefficients
conf_int_house <- confint(house_model, level = 0.95)
print(conf_int_house)

# Step 3: Check Model Assumptions
# Create diagnostic plots
par(mfrow = c(2, 2))
plot(house_model)
par(mfrow = c(1, 1))

# Test for normality of residuals
shapiro_house <- shapiro.test(residuals(house_model))
print(shapiro_house)

# Test for homoscedasticity
bp_house <- bptest(house_model)  # Breusch-Pagan test
print(bp_house)

# Step 4: Make Predictions
# Create new data for prediction
new_houses <- data.frame(size_sqft = c(1500, 2200, 2800))

# Make predictions
predictions_house <- predict(house_model, newdata = new_houses, interval = "prediction")
prediction_results <- cbind(new_houses, predictions_house)
print(prediction_results)

# Visualize the regression line with confidence and prediction intervals
house_reg_plot <- ggplot(house_data, aes(x = size_sqft, y = price_thousands)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", formula = y ~ x, color = "red", fill = "pink") +
  labs(title = "Simple Linear Regression: House Price vs. Size",
       x = "House Size (square feet)",
       y = "House Price (thousands $)") +
  theme_minimal()
print(house_reg_plot)

# ===== Example 2: Multiple Linear Regression (House Prices with Multiple Predictors) =====
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

# Step 1: Explore the Data
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
cor_house_multi <- cor(house_data_multi)
print(cor_house_multi)

# Step 2: Fit a Multiple Linear Regression Model
# Fit a multiple linear regression model
house_model_multi <- lm(price_thousands ~ size_sqft + bedrooms + age_years, 
                        data = house_data_multi)

# View the model summary
summary(house_model_multi)

# Step 3: Check for Multicollinearity
# Calculate Variance Inflation Factors (VIF)
vif_result <- vif(house_model_multi)
print(vif_result)

# Step 4: Check Model Assumptions
# Create diagnostic plots
par(mfrow = c(2, 2))
plot(house_model_multi)
par(mfrow = c(1, 1))

# Step 5: Make Predictions
# Create new data for prediction
new_houses_multi <- data.frame(
  size_sqft = c(1500, 2200, 2800),
  bedrooms = c(3, 4, 5),
  age_years = c(10, 5, 15)
)

# Make predictions
predictions_multi <- predict(house_model_multi, newdata = new_houses_multi, 
                            interval = "prediction")
prediction_results_multi <- cbind(new_houses_multi, predictions_multi)
print(prediction_results_multi)

# Model Selection in Multiple Regression
# Fit a full model
full_model <- lm(price_thousands ~ size_sqft + bedrooms + age_years, 
                data = house_data_multi)

# Perform stepwise selection
step_model <- step(full_model, direction = "both", trace = TRUE)

# View the final model
summary(step_model)

# Comparing Models with AIC
# Fit different models
model1 <- lm(price_thousands ~ size_sqft, data = house_data_multi)
model2 <- lm(price_thousands ~ size_sqft + bedrooms, data = house_data_multi)
model3 <- lm(price_thousands ~ size_sqft + bedrooms + age_years, data = house_data_multi)

# Compare AIC values
aic_comparison <- AIC(model1, model2, model3)
print(aic_comparison)

# ===== Example 3: Polynomial Regression (Non-linear Relationship) =====
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
nonlinear_scatter <- ggplot(nonlinear_data, aes(x = x, y = y)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  labs(title = "Non-linear Relationship",
       x = "X",
       y = "Y") +
  theme_minimal()
print(nonlinear_scatter)

# Step 1: Fit a Linear Model (for comparison)
# Fit a simple linear model
linear_model <- lm(y ~ x, data = nonlinear_data)

# View the model summary
summary(linear_model)

# Step 2: Fit a Polynomial Model
# Fit a quadratic model (polynomial of degree 2)
quadratic_model <- lm(y ~ x + I(x^2), data = nonlinear_data)

# View the model summary
summary(quadratic_model)

# Fit a cubic model (polynomial of degree 3)
cubic_model <- lm(y ~ x + I(x^2) + I(x^3), data = nonlinear_data)

# View the model summary
summary(cubic_model)

# Step 3: Compare Models
# Compare models using ANOVA
anova_comparison <- anova(linear_model, quadratic_model, cubic_model)
print(anova_comparison)

# Compare models using AIC
aic_poly_comparison <- AIC(linear_model, quadratic_model, cubic_model)
print(aic_poly_comparison)

# Step 4: Visualize the Models
# Create a grid of x values for prediction
grid <- data.frame(x = seq(min(nonlinear_data$x), max(nonlinear_data$x), length.out = 100))

# Make predictions for each model
grid$linear <- predict(linear_model, newdata = grid)
grid$quadratic <- predict(quadratic_model, newdata = grid)
grid$cubic <- predict(cubic_model, newdata = grid)

# Plot the data and model predictions
poly_comparison_plot <- ggplot(nonlinear_data, aes(x = x, y = y)) +
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
print(poly_comparison_plot)

# Alternative Approach: Using the poly() Function
# Fit polynomial models using poly()
poly_model2 <- lm(y ~ poly(x, 2), data = nonlinear_data)  # Degree 2
poly_model3 <- lm(y ~ poly(x, 3), data = nonlinear_data)  # Degree 3

# View the model summaries
summary(poly_model2)
summary(poly_model3)

# ===== Example 4: Generalized Additive Models (GAMs) =====
# Use the same non-linear dataset and fit a GAM
# Fit a GAM
gam_model <- gam(y ~ s(x), data = nonlinear_data)

# View the model summary
summary(gam_model)

# Step 1: Visualize the GAM
# Plot the GAM
plot(gam_model, residuals = TRUE, pch = 19, cex = 1)

# Create a grid of x values for prediction
grid$gam <- predict(gam_model, newdata = grid)

# Plot the data and GAM prediction
gam_plot <- ggplot(nonlinear_data, aes(x = x, y = y)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  geom_line(data = grid, aes(y = gam), color = "red", size = 1) +
  labs(title = "Generalized Additive Model (GAM)",
       x = "X",
       y = "Y") +
  theme_minimal()
print(gam_plot)

# Step 2: Compare GAM with Polynomial Models
# Plot the data and model predictions
gam_poly_comparison <- ggplot(nonlinear_data, aes(x = x, y = y)) +
  geom_point(color = "blue", size = 3, alpha = 0.7) +
  geom_line(data = grid, aes(y = quadratic, color = "Polynomial"), size = 1) +
  geom_line(data = grid, aes(y = gam, color = "GAM"), size = 1) +
  scale_color_manual(name = "Model", 
                     values = c("Polynomial" = "green", "GAM" = "red")) +
  labs(title = "Comparison of Polynomial and GAM Models",
       x = "X",
       y = "Y") +
  theme_minimal()
print(gam_poly_comparison)

# Step 3: Check GAM Assumptions
# Check GAM residuals
gam.check(gam_model)

# GAMs with Multiple Predictors
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

# ===== Example 5: Logistic Regression (Exam Pass/Fail) =====
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

# Step 1: Explore the Data
# Create scatter plots
exam_scatter <- ggplot(exam_data, aes(x = study_hours, y = prev_gpa, color = pass)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = c("Fail" = "red", "Pass" = "green")) +
  labs(title = "Exam Outcome by Study Hours and Previous GPA",
       x = "Study Hours",
       y = "Previous GPA",
       color = "Outcome") +
  theme_minimal()
print(exam_scatter)

# Step 2: Fit a Logistic Regression Model
# Fit a logistic regression model
logistic_model <- glm(pass ~ study_hours + prev_gpa, 
                      data = exam_data, 
                      family = binomial)

# View the model summary
summary(logistic_model)

# Step 3: Interpret the Coefficients
# Extract coefficients
coef_logistic <- coef(logistic_model)
print(coef_logistic)

# Convert log-odds to odds ratios
odds_ratios <- exp(coef(logistic_model))
print(odds_ratios)

# Confidence intervals for odds ratios
ci_odds_ratios <- exp(confint(logistic_model))
print(ci_odds_ratios)

# Step 4: Assess Model Fit
# Pseudo R-squared (McFadden)
null_model <- glm(pass ~ 1, data = exam_data, family = binomial)
pseudo_r2 <- 1 - logLik(logistic_model)/logLik(null_model)
print(pseudo_r2)

# AIC
aic_logistic <- AIC(logistic_model)
print(aic_logistic)

# Classification table
predicted_probs <- predict(logistic_model, type = "response")
predicted_classes <- ifelse(predicted_probs > 0.5, "Pass", "Fail")
confusion_matrix <- table(Predicted = predicted_classes, Actual = exam_data$pass)
print(confusion_matrix)

# Calculate accuracy
accuracy <- mean(predicted_classes == exam_data$pass)
print(paste("Accuracy:", round(accuracy, 3)))

# Step 5: Visualize the Decision Boundary
# Create a grid of values for prediction
grid_x1 <- seq(min(exam_data$study_hours), max(exam_data$study_hours), length.out = 100)
grid_x2 <- seq(min(exam_data$prev_gpa), max(exam_data$prev_gpa), length.out = 100)
grid_logistic <- expand.grid(study_hours = grid_x1, prev_gpa = grid_x2)

# Make predictions on the grid
grid_logistic$prob <- predict(logistic_model, newdata = grid_logistic, type = "response")
grid_logistic$predicted <- ifelse(grid_logistic$prob > 0.5, "Pass", "Fail")

# Plot the decision boundary
decision_boundary_plot <- ggplot() +
  geom_tile(data = grid_logistic, aes(x = study_hours, y = prev_gpa, fill = prob), alpha = 0.3) +
  geom_contour(data = grid_logistic, aes(x = study_hours, y = prev_gpa, z = prob), 
               breaks = 0.5, color = "black", size = 1) +
  geom_point(data = exam_data, aes(x = study_hours, y = prev_gpa, color = pass), 
             size = 3, alpha = 0.7) +
  scale_fill_gradient(low = "blue", high = "red") +
  scale_color_manual(values = c("Fail" = "red", "Pass" = "green")) +
  labs(title = "Logistic Regression Decision Boundary",
       x = "Study Hours",
       y = "Previous GPA",
       color = "Actual Outcome",
       fill = "Probability of Pass") +
  theme_minimal()
print(decision_boundary_plot)

# ROC Curve and AUC
# Calculate ROC curve
if (!requireNamespace("pROC", quietly = TRUE)) {
  install.packages("pROC")
}
library(pROC)
roc_obj <- roc(exam_data$pass, predicted_probs)

# Plot ROC curve
plot(roc_obj, main = "ROC Curve", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")

# Calculate AUC
auc_value <- auc(roc_obj)
print(paste("AUC:", round(auc_value, 3)))

# ===== Model Validation and Cross-Validation =====
# Set up 10-fold cross-validation for the house price model
set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)

# Train a linear regression model with cross-validation
cv_model <- train(price_thousands ~ size_sqft + bedrooms + age_years, 
                 data = house_data_multi,
                 method = "lm",
                 trControl = ctrl)

# View the results
print(cv_model)
summary(cv_model)

# ===== Generate a Larger Dataset for Practice =====
# Create a dataset for real estate analysis
set.seed(123)  # For reproducibility
n_houses <- 200

# Generate house features
real_estate_data <- data.frame(
  id = 1:n_houses,
  size_sqft = round(runif(n_houses, 800, 4000)),
  bedrooms = sample(1:6, n_houses, replace = TRUE, prob = c(0.05, 0.15, 0.35, 0.30, 0.10, 0.05)),
  bathrooms = sample(1:4, n_houses, replace = TRUE, prob = c(0.15, 0.45, 0.30, 0.10)),
  age_years = round(runif(n_houses, 0, 50)),
  lot_size_sqft = round(runif(n_houses, 2000, 20000)),
  garage_cars = sample(0:3, n_houses, replace = TRUE, prob = c(0.10, 0.30, 0.50, 0.10)),
  has_pool = sample(c(0, 1), n_houses, replace = TRUE, prob = c(0.85, 0.15)),
  has_fireplace = sample(c(0, 1), n_houses, replace = TRUE, prob = c(0.70, 0.30)),
  neighborhood = factor(sample(c("Urban", "Suburban", "Rural"), n_houses, replace = TRUE, 
                              prob = c(0.35, 0.50, 0.15))),
  school_rating = sample(1:10, n_houses, replace = TRUE),
  price = numeric(n_houses)
)

# Generate prices based on features with some non-linear relationships
for (i in 1:nrow(real_estate_data)) {
  # Base price
  base_price <- 100000
  
  # Size effect (non-linear)
  size_effect <- 100 * real_estate_data$size_sqft[i] + 0.01 * real_estate_data$size_sqft[i]^2
  
  # Bedroom effect
  bedroom_effect <- 15000 * real_estate_data$bedrooms[i]
  
  # Bathroom effect
  bathroom_effect <- 25000 * real_estate_data$bathrooms[i]
  
  # Age effect (non-linear)
  age_effect <- -2000 * sqrt(real_estate_data$age_years[i])
  
  # Lot size effect
  lot_effect <- 2 * real_estate_data$lot_size_sqft[i]
  
  # Garage effect
  garage_effect <- 20000 * real_estate_data$garage_cars[i]
  
  # Pool effect
  pool_effect <- 30000 * real_estate_data$has_pool[i]
  
  # Fireplace effect
  fireplace_effect <- 10000 * real_estate_data$has_fireplace[i]
  
  # Neighborhood effect
  if (real_estate_data$neighborhood[i] == "Urban") {
    neighborhood_effect <- 50000
  } else if (real_estate_data$neighborhood[i] == "Suburban") {
    neighborhood_effect <- 30000
  } else {
    neighborhood_effect <- 10000
  }
  
  # School rating effect
  school_effect <- 10000 * real_estate_data$school_rating[i]
  
  # Calculate total price with random noise
  real_estate_data$price[i] <- base_price + size_effect + bedroom_effect + bathroom_effect + 
    age_effect + lot_effect + garage_effect + pool_effect + fireplace_effect + 
    neighborhood_effect + school_effect + rnorm(1, 0, 50000)
}

# Ensure prices are positive and round to nearest thousand
real_estate_data$price <- round(pmax(real_estate_data$price, 50000), -3)

# View the first few rows of the dataset
head(real_estate_data)

# Summary statistics
summary(real_estate_data)

# Explore relationships
# Size vs. Price
ggplot(real_estate_data, aes(x = size_sqft, y = price)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(title = "House Price vs. Size",
       x = "Size (sq ft)",
       y = "Price ($)") +
  theme_minimal()

# Age vs. Price
ggplot(real_estate_data, aes(x = age_years, y = price)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  labs(title = "House Price vs. Age",
       x = "Age (years)",
       y = "Price ($)") +
  theme_minimal()

# Boxplot of Price by Neighborhood
ggplot(real_estate_data, aes(x = neighborhood, y = price, fill = neighborhood)) +
  geom_boxplot() +
  labs(title = "House Price by Neighborhood",
       x = "Neighborhood",
       y = "Price ($)") +
  theme_minimal()

# Fit a multiple linear regression model
lm_real_estate <- lm(price ~ size_sqft + bedrooms + bathrooms + age_years + 
                     lot_size_sqft + garage_cars + has_pool + has_fireplace + 
                     neighborhood + school_rating, 
                   data = real_estate_data)

# View the model summary
summary(lm_real_estate)

# Try a model with polynomial terms for size and age
poly_real_estate <- lm(price ~ poly(size_sqft, 2) + bedrooms + bathrooms + 
                      poly(age_years, 2) + lot_size_sqft + garage_cars + 
                      has_pool + has_fireplace + neighborhood + school_rating, 
                    data = real_estate_data)

# View the model summary
summary(poly_real_estate)

# Compare models
anova(lm_real_estate, poly_real_estate)
AIC(lm_real_estate, poly_real_estate)

# Save the dataset for future use
write.csv(real_estate_data, "real_estate_data.csv", row.names = FALSE)

# Print a message indicating the script has completed
cat("Module 5 script has completed successfully!\n")
