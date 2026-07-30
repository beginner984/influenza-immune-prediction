# R/05_train_elastic_net.R
# Train and internally validate an Elastic Net model.
# Only the TRAINING data are used in this script.
# The test set remains untouched.
install.packages("glmnet")
library(caret)
library(glmnet)
library(dplyr)

# ------------------------------------------------------------
# 1. File locations
# ------------------------------------------------------------

training_file <- "data/processed/training_data_16.csv"
model_dir <- "models"
result_dir <- "results"

dir.create(model_dir, showWarnings = FALSE)
dir.create(result_dir, showWarnings = FALSE)

# ------------------------------------------------------------
# 2. Read the training data
# ------------------------------------------------------------

training_data <- read.csv(
  training_file,
  stringsAsFactors = FALSE,
  check.names = TRUE
)

# Keep Positive as the event class.
training_data$PCRpositive <- factor(
  training_data$PCRpositive,
  levels = c("Positive", "Negative")
)

# Participant ID is not a predictor.
training_model <- training_data %>%
  select(-Study.ID)

# ------------------------------------------------------------
# 3. Create stratified repeated cross-validation folds
# ------------------------------------------------------------

# Five folds are used because there are only 28 positive cases.
# Repeating the process five times gives a more stable estimate.
set.seed(2026)

cv_folds <- createMultiFolds(
  training_model$PCRpositive,
  k = 5,
  times = 5
)

# Save the folds so Random Forest and XGBoost can use exactly
# the same resampling partitions later.
saveRDS(
  cv_folds,
  file.path(model_dir, "cross_validation_folds.rds")
)

train_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5,
  index = cv_folds,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  returnResamp = "final"
)

# ------------------------------------------------------------
# 4. Define the Elastic Net tuning grid
# ------------------------------------------------------------

# alpha:
#   0 = ridge regression
#   1 = lasso regression
#   values between 0 and 1 mix both penalties
#
# lambda controls the strength of regularisation.
elastic_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),
  lambda = 10^seq(-4, 0, length.out = 30)
)

# ------------------------------------------------------------
# 5. Train the Elastic Net model
# ------------------------------------------------------------

set.seed(2026)

elastic_net_model <- train(
  PCRpositive ~ .,
  data = training_model,
  method = "glmnet",
  metric = "ROC",
  tuneGrid = elastic_grid,
  trControl = train_control,

  # Scaling is estimated separately inside each training fold.
  # This prevents information leakage from validation folds.
  preProcess = c("center", "scale")
)

# ------------------------------------------------------------
# 6. Save the fitted model and tuning results
# ------------------------------------------------------------

saveRDS(
  elastic_net_model,
  file.path(model_dir, "elastic_net_model.rds")
)

write.csv(
  elastic_net_model$results,
  file.path(result_dir, "elastic_net_tuning_results.csv"),
  row.names = FALSE
)

write.csv(
  elastic_net_model$bestTune,
  file.path(result_dir, "elastic_net_best_tuning.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. Extract coefficients at the selected lambda
# ------------------------------------------------------------

coefficient_matrix <- as.matrix(
  coef(
    elastic_net_model$finalModel,
    s = elastic_net_model$bestTune$lambda
  )
)

coefficient_table <- data.frame(
  variable = rownames(coefficient_matrix),
  coefficient = as.numeric(coefficient_matrix[, 1]),
  row.names = NULL
) %>%
  mutate(selected = coefficient != 0) %>%
  arrange(desc(abs(coefficient)))

write.csv(
  coefficient_table,
  file.path(result_dir, "elastic_net_coefficients.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 8. Print the main results
# ------------------------------------------------------------

best_row <- elastic_net_model$results %>%
  filter(
    alpha == elastic_net_model$bestTune$alpha,
    lambda == elastic_net_model$bestTune$lambda
  )

cat("\nElastic Net training completed.\n\n")

cat("Best tuning values:\n")
print(elastic_net_model$bestTune)

cat("\nRepeated 5-fold cross-validation performance:\n")
print(best_row[, c("ROC", "Sens", "Spec")])

cat("\nNon-zero coefficients:\n")
print(coefficient_table %>% filter(selected))

cat("\nThe test set has not been used.\n")
