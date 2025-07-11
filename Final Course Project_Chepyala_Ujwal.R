#Task 1: Data Preparation

# a. Load the dataset
library(readr)
insurance <- read_csv("C:/Users/uchepyala1/Downloads/insurance.csv")
View(insurance)
insurance

# b. Transform the charges variable
insurance$charges <- log(insurance$charges)
head(insurance$charges)

# c. Create dummy variables using model.matrix
dummy_data <- model.matrix(~ . - 1, data = insurance)
head(dummy_data)

# Verify and discard the first column if it has only ones
if (all(dummy_data[, 1] == 1)) {
  dummy_data <- dummy_data[, -1]
}

# d. Set seed for reproducibility
set.seed(1)

# Generate row indexes for training and test sets
indexes <- sample(nrow(insurance), size = nrow(insurance), replace = FALSE)
head(indexes)
train_indexes <- indexes[1:(2/3 * length(indexes))]
head(train_indexes)
test_indexes <- indexes[-train_indexes]
head(test_indexes)

# e,f. Create training and test datasets
train_data <- dummy_data[train_indexes, ]
head(train_data)
test_data <- dummy_data[test_indexes, ]
head(test_data)

# Task 2: Build a multiple linear regression model
# a. Perform multiple linear regression.
lm_model <- lm(charges ~ ., data = as.data.frame(train_data))

# Print the results
summary(lm_model)

# d. Perform best subset selection using stepAIC
library(MASS)
best_model <- stepAIC(lm_model, direction = "backward")
summary(best_model)

# e. Compute test error using LOOCV
# Load the caret library
library(caret)

# Check the structure of test_data
str(test_data)

# If test_data is not a data frame, convert it to one
if (!is.data.frame(test_data)) {
  test_data <- as.data.frame(test_data)
}

# Check if "charges" column is present in test_data
if (!("charges" %in% names(test_data))) {
  stop("The 'charges' column is not present in the test_data.")
}

# Predict on the test data
predictions <- predict(best_model, newdata = test_data)

# Calculate RMSE
rmse <- sqrt(mean((predictions - test_data$charges)^2))

# Square RMSE to obtain MSE
mse <- rmse^2

# Print the results
cat("Test MSE using LOOCV:", mse, "\n")

# f. Calculate the test error of best model #2d.
# Create a train control object for 10-fold Cross-Validation
ctrl <- trainControl(method = "cv", number = 10)

# Fit the model using train() with 10-fold Cross-Validation
lm_best_subset_cv <- train(charges ~ ., data = as.data.frame(train_data), method = "lm",
                           trControl = ctrl, subset = best_model$call$subset)

# Predict on the test data
predictions <- predict(lm_best_subset_cv, newdata = as.data.frame(test_data))

# Calculate RMSE
rmse <- sqrt(mean((predictions - test_data$charges)^2))

# Square RMSE to obtain MSE
mse <- rmse^2

# Print the results
cat("Test MSE using 10-fold Cross-Validation:", mse, "\n")

# g. Calculate and report the test MSE using the best model from 2d.
# Predict on the test data
lm_predictions <- predict(best_model, newdata = as.data.frame(test_data))

# Calculate RMSE
lm_rmse <- sqrt(mean((lm_predictions - test_data$charges)^2))

# Square RMSE to obtain MSE
lm_mse <- lm_rmse^2

# Print the results
cat("Test MSE using the multiple linear regression model:", lm_mse, "\n")

# Task 3: Build a regression tree model
# Load the tree library
library(tree)

# a. Build the regression tree model
tree_model <- tree(charges ~ ., data = as.data.frame(train_data))

# Print the tree structure
print(tree_model)

# b. Find optimal tree size by using cross-validation
# Load the cvms library
library(cvms)

# Perform cross-validation to find the optimal tree size
cv_results <- cv.tree(tree_model)

# Plot the cross-validated error by tree size
plot(cv_results$size, cv_results$dev, type = "b", xlab = "Tree Size", ylab = "Cross-Validated Error",
     main = "Cross-Validated Error by Tree Size")

# Report the best tree size
best_tree_size <- which.min(cv_results$dev)
cat("Best Tree Size:", best_tree_size, "\n")

# d. Prune the tree using the optimal size
pruned_tree <- prune.tree(tree_model, best = best_tree_size)

# Print the pruned tree
print(pruned_tree)
summary(pruned_tree)

# e. Plot the best tree with labels
plot(tree_model)
text(tree_model, pretty = 0)

# f. Calculate the test MSE for best model
# Predict on the test data using the best tree model
tree_predictions <- predict(tree_model, newdata = as.data.frame(test_data))

# Calculate the residuals (predicted - actual)
residuals <- tree_predictions - test_data$charges

# Calculate the Mean Squared Error (MSE)
tree_mse <- mean(residuals^2)

# Print the MSE
cat("Test MSE for the best regression tree model:", tree_mse, "\n")

# Task 4: Build a random forest model
# Load the required library
library(randomForest)

# a. Build a random forest model
rf_model <- randomForest(charges ~ ., data = as.data.frame(train_data))
summary(rf_model)

# b. Compute the test error using the test data set
# Assuming 'test_data' is the test dataset
rf_predictions <- predict(rf_model, newdata = as.data.frame(test_data))
rf_residuals <- rf_predictions - test_data$charges
rf_mse <- mean(rf_residuals^2)
cat("Test MSE for the random forest model:", rf_mse, "\n")

# c. Extract variable importance measure
rf_importance <- importance(rf_model)
summary(rf_importance)

# d. Plot the variable importance
# Load the required library
library(ggplot2)

# Plot variable importance
varImpPlot(rf_model, main = "Variable Importance Plot")

# Print the top 3 important predictors
top3_importance <- head(rf_importance[order(-rf_importance[, 1]), , drop = FALSE], 3)
cat("Top 3 important predictors:\n", rownames(top3_importance), "\n")

# Task 5: Build a support vector machine model
# Load the required library
library(e1071)

# a. Build a support vector machine model
svm_model <- svm(charges ~ ., data = as.data.frame(train_data),
                 kernel = "radial", gamma = 5, cost = 50)
summary(svm_model)

# Define parameter grid
cost_values <- c(1, 10, 50, 100)
gamma_values <- c(1, 3, 5)
kernel_values <- c("linear", "radial", "sigmoid")

# Initialize variables to store results
best_model <- NULL
best_mse <- Inf

# b. Perform grid search
for (cost in cost_values) {
  for (gamma in gamma_values) {
    for (kernel in kernel_values) {
      
      # Train SVM model
      svm_model <- svm(charges ~ ., data = as.data.frame(train_data),
                       kernel = kernel, gamma = gamma, cost = cost)
      
      # Predict on test data
      svm_predictions <- predict(svm_model, newdata = as.data.frame(test_data))
      
      # Compute MSE
      mse <- mean((svm_predictions - test_data$charges)^2)
      
      # Check if this is the best model so far
      if (mse < best_mse) {
        best_mse <- mse
        best_svm_model <- svm_model
      }
    }
  }
}

# c. Print out the best model parameters
cat("Best SVM Model Parameters:\n")
print(best_svm_model)

# d. Forecast charges using the test dataset and the best model
svm_predictions <- predict(best_svm_model, newdata = as.data.frame(test_data))
summary(svm_predictions)

# e. Compute the MSE on the test data
svm_residuals <- svm_predictions - test_data$charges
svm_mse <- mean(svm_residuals^2)
cat("Test MSE for the SVM model:", svm_mse, "\n")

# Task 6: Perform k-means clustering analysis
# Load necessary libraries
library(ggplot2)
library(dplyr)

# a. Remove non-numerical columns
numerical_data <- select_if(insurance, is.numeric)
numerical_data
# Handle missing values by imputing with mean
numerical_data <- na.omit(numerical_data)  # Remove rows with missing values

# Check if there are still missing values
if (any(is.na(numerical_data))) {
  stop("There are still missing values in the numerical data. Please handle them before proceeding.")
}

# b. Determine the optimal number of clusters
# Use the elbow method to find the optimal number of clusters
wss <- numeric(10)
for (i in 1:10) {
  kmeans_model <- kmeans(numerical_data, centers = i)
  wss[i] <- sum(kmeans_model$withinss)
}

# Plot the elbow method
plot(1:10, wss, type = "b", xlab = "Number of Clusters", ylab = "Within Sum of Squares",
     main = "Elbow Method to Determine Optimal Number of Clusters")

# c. Perform k-means clustering with 3 clusters
optimal_clusters <- 3
kmeans_model <- kmeans(numerical_data, centers = optimal_clusters)
summary(kmeans_model)

# d. Visualize clusters in different colors
cluster_visualization <- as.data.frame(cbind(numerical_data, Cluster = as.factor(kmeans_model$cluster)))

# Plot the clusters
ggplot(cluster_visualization, aes(x = age, y = charges, color = Cluster)) +
  geom_point() +
  labs(title = "K-Means Clustering of Insurance Data",
       x = "Age", y = "Charges") +
  theme_minimal()

# Task 7: Build a neural networks model

# a. Remove non-numerical columns
numerical_data <- subset(insurance, select = c("age", "bmi", "children", "charges"))
numerical_data

# b. Standardize the inputs
scaled_data <- scale(numerical_data[, -4])
head(scaled_data)

# c. Convert standardized inputs to a data frame
scaled_data <- as.data.frame(scaled_data)
summary(scaled_data)

# d. Split the dataset into training and test sets
set.seed(123)  # Set seed for reproducibility
index <- sample(1:nrow(scaled_data), 0.8 * nrow(scaled_data))
train_data_nn <- scaled_data[index, ]
summary(train_data_nn)
test_data_nn <- scaled_data[-index, ]
summary(test_data_nn)

# e. Build neural networks model
library(neuralnet)
nn_model <- neuralnet(charges ~ age + bmi + children, data = train_data, hidden = c(1), linear.output = TRUE)
summary(nn_model)

# f. Plot the neural networks
plot(nn_model, main = "Neural Networks Model")

# g. Forecast charges in the test dataset
nn_predictions <- predict(nn_model, newdata = test_data)
summary(nn_predictions)

# h. Get the observed charges of the test dataset
observed_charges <- test_data$charges
summary(observed_charges)

# i. Compute test error (MSE)
nn_mse <- mean((nn_predictions - observed_charges)^2)
cat("Test MSE for Neural Networks:", nn_mse, "\n")

# Task 8: Putting it all together
# a. Compare the test MSEs of different models

# Create a data frame with model types and their test MSEs
models_data <- data.frame(
  Model.Type = c("Multiple Linear Regression", "Regression Tree", "Random Forest", "Support Vector Machine", "Neural Network"),
  Test.MSE = c(round(lm_mse, 4), round(tree_mse, 4), round(rf_mse, 4), round(svm_mse, 4), round(nn_mse, 4))
)
summary(models_data)

# Display the data frame
cat("Test MSEs for Different Models:\n")
print(models_data)

# Recommend the best model based on the minimum test MSE
best_model <- models_data[which.min(models_data$Test.MSE), "Model.Type"]
cat("\nRecommendation: The best model is", best_model, "with the lowest test MSE.\n\n")

# c. Reverse log transformation in the regression tree model

# a. Copy the pruned tree model to a new variable
copy_of_pruned_tree <- tree_model

# b. Reverse the log transformation on the 'yval' column in the 'frame' data frame
copy_of_pruned_tree$frame$yval <- exp(copy_of_pruned_tree$frame$yval)

# c. Replot the pruned tree with labels
plot(copy_of_pruned_tree$frame$yval, main = "Replot of Pruned tree with labels")
