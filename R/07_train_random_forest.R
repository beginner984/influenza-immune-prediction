# R/07_train_random_forest.R
# Train and internally validate a Random Forest model.
# It uses the SAME training participants and SAME cross-validation
# folds as the Elastic Net model.
# The external test set remains untouched.
install.packages("ranger")
library(caret)
library(ranger)
library(dplyr)

# ------------------------------------------------------------
# 1. File locations
# ------------------------------------------------------------

training_file <- "data/processed/training_data_16.csv"
folds_file <- "models/cross_validation_folds.rds"
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

# Positive is the symptomatic-influenza event.
training_data$PCRpositive <- factor(
  training_data$PCRpositive,
  levels = c("Positive", "Negative")
)

# Participant ID must not enter the model.
training_model <- training_data %>%
  select(-Study.ID)

# ------------------------------------------------------------
# 3. Load the exact folds used for Elastic Net
# ------------------------------------------------------------

cv_folds <- readRDS(folds_file)

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
# 4. Define a manageable Random Forest tuning grid
# ------------------------------------------------------------

# mtry = number of predictors considered at each split
# splitrule = method used to choose the split
# min.node.size = minimum number of observations in a terminal node
rf_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 12, 16),
  splitrule = c("gini", "extratrees"),
  min.node.size = c(1, 3, 5, 10)
)

# ------------------------------------------------------------
# 5. Train the Random Forest
# ------------------------------------------------------------

set.seed(2026)

random_forest_model <- train(
  PCRpositive ~ .,
  data = training_model,
  method = "ranger",
  metric = "ROC",
  tuneGrid = rf_grid,
  trControl = train_control,

  # A large number of trees gives stable predictions.
  num.trees = 1000,

  # Permutation importance is easier to interpret than impurity importance.
  importance = "permutation"
)

# ------------------------------------------------------------
# 6. Save model and tuning results
# ------------------------------------------------------------

saveRDS(
  random_forest_model,
  file.path(model_dir, "random_forest_model.rds")
)

write.csv(
  random_forest_model$results,
  file.path(result_dir, "random_forest_tuning_results.csv"),
  row.names = FALSE
)

write.csv(
  random_forest_model$bestTune,
  file.path(result_dir, "random_forest_best_tuning.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. Calculate variable importance
# ------------------------------------------------------------

rf_importance <- varImp(
  random_forest_model,
  scale = TRUE
)$importance %>%
  tibble::rownames_to_column("variable") %>%
  arrange(desc(Overall))

write.csv(
  rf_importance,
  file.path(result_dir, "random_forest_variable_importance.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 8. Print the main cross-validation results
# ------------------------------------------------------------

best_row <- random_forest_model$results %>%
  filter(
    mtry == random_forest_model$bestTune$mtry,
    splitrule == random_forest_model$bestTune$splitrule,
    min.node.size == random_forest_model$bestTune$min.node.size
  )

cat("\nRandom Forest training completed.\n\n")

cat("Best tuning values:\n")
print(random_forest_model$bestTune)

cat("\nRepeated 5-fold cross-validation performance:\n")
print(best_row[, c("ROC", "Sens", "Spec")])

cat("\nTop 10 variables by permutation importance:\n")
print(head(rf_importance, 10))

cat("\nThe external test set has not been used.\n")
