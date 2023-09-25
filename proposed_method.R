# setwd("C:/Users/QXZ1DJT/Desktop")
library(readr)  # For reading CSV files
library(dplyr)  # For data manipulation
library(glmnet) # For logistic regression
library(boot)

X <- read.csv("LogRegX.csv")  # Replace "X.csv" with the path to your X file
y <- read.csv("LogRegy.csv")  # Replace "y.csv" with the path to your y file


merged_data <- merge(X, y, by = "X")
merged_data <- merged_data %>% select(-X)
merged_data$target <- as.factor(merged_data$X0)
merged_data <- merged_data %>% select(-X0)

# 1. Compute the sum of the first 10 columns for each row.
#sum_first_10 <- rowSums(merged_data[, 1:10])

# 2. Compute the sum of the last 9 columns for each row.
#sum_last <- rowSums(merged_data[, (ncol(merged_data)-8):ncol(merged_data)-1])

# 3. Divide each of the first 10 columns by the sum from step 1.
#merged_data[, 1:10] <- sweep(merged_data[, 1:10], 1, sum_first_10, FUN = "/")

# 4. Divide each of the last 9 columns by the sum from step 2.
#merged_data[, (ncol(merged_data)-8):ncol(merged_data)-1] <- sweep(merged_data[, (ncol(merged_data)-8):ncol(merged_data)-1], 1, sum_last, FUN = "/")


set.seed()  # For reproducibility
train_ratio <- 0.8  # 80% of the data for training, adjust as needed

# Split the data into training and testing sets
sample_index <- sample(seq_len(nrow(merged_data)), size = floor(train_ratio * nrow(merged_data)))
train_data <- merged_data[sample_index, ]
test_data <- merged_data[-sample_index, ]


model <- glm(target ~ ., data = train_data, family = "binomial")
#model <- glmnet(train_data %>% select(-target), train_data$target, alpha = alpha, lambda = lambda, family = "binomial")
predictions <- as.integer(predict(model, newdata = test_data, type = "response") > 0.5)


accuracy <- mean(predictions == test_data$target)

# Print the accuracy
cat("Accuracy:", accuracy, "\n")

# Cross validation
library(caret)
library(glmnet)

# Assuming X and y are data.frame and vector respectively

# Create a 5-fold cross-validation object
set.seed(42)
kf <- createFolds(merged_data$target, k = 5, list = TRUE, returnTrain = TRUE)
accs <- list("Logistic Regression" = numeric())

# Loop through each fold
for (fold in kf) {
  # Subset the data for this fold
  X_train <- merged_data[fold, , drop = FALSE]
  X_test <- merged_data[-fold, , drop = FALSE] %>% select(-target)
  y_test <- merged_data[-fold,] %>% select(target)
  
  # Train logistic regression
  model = glm(target ~ ., data = X_train, family = "binomial")
  # Predict and calculate accuracy
  predictions <- as.integer(predict(model, newdata = X_test, type = "response") > 0.5)
  accuracy <- mean(predictions == y_test)
  
  accs$`Logistic Regression` <- c(accs$`Logistic Regression`, accuracy)
}

# Display accuracies
print(mean(accs$`Logistic Regression`))

library(xtable)
latex_code = xtable(summary(model)$coefficients, digits=4)
print(latex_code)

