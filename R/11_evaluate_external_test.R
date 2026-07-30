# R/11_evaluate_external_test.R
# Final evaluation of Elastic Net, Random Forest, XGBoost,
# and the equal-weight ensemble on the untouched external test set.
#
# IMPORTANT:
# All models, hyperparameters, and classification thresholds were selected
# using training data only. Do not retune models after viewing these results.

library(caret)
library(xgboost)
library(dplyr)
library(pROC)

# ------------------------------------------------------------
# 1. File locations
# ------------------------------------------------------------

test_file <- "data/processed/testing_data_16.csv"

elastic_model_file <- "models/elastic_net_model.rds"
rf_model_file <- "models/random_forest_model.rds"
xgb_model_file <- "models/xgboost_native_model.json"
xgb_metadata_file <- "models/xgboost_native_metadata.rds"

dir.create("results", showWarnings = FALSE)

# ------------------------------------------------------------
# 2. Read untouched test data
# ------------------------------------------------------------

test_data <- read.csv(
  test_file,
  stringsAsFactors = FALSE,
  check.names = TRUE
)

required_columns <- c("Study.ID", "PCRpositive")

if (!all(required_columns %in% names(test_data))) {
  stop("The test file is missing Study.ID or PCRpositive.")
}

test_data$PCRpositive <- factor(
  test_data$PCRpositive,
  levels = c("Positive", "Negative")
)

test_predictors <- test_data %>%
  select(-Study.ID, -PCRpositive)

# ------------------------------------------------------------
# 3. Load fitted models
# ------------------------------------------------------------

elastic_model <- readRDS(elastic_model_file)
rf_model <- readRDS(rf_model_file)

xgb_model <- xgb.load(xgb_model_file)
xgb_metadata <- readRDS(xgb_metadata_file)

# ------------------------------------------------------------
# 4. Generate test probabilities
# ------------------------------------------------------------

elastic_probability <- predict(
  elastic_model,
  newdata = test_predictors,
  type = "prob"
)[, "Positive"]

rf_probability <- predict(
  rf_model,
  newdata = test_predictors,
  type = "prob"
)[, "Positive"]

# Preserve exactly the predictor order used to train XGBoost.
missing_xgb_predictors <- setdiff(
  xgb_metadata$predictors,
  names(test_data)
)

if (length(missing_xgb_predictors) > 0) {
  stop(
    "Missing XGBoost predictors: ",
    paste(missing_xgb_predictors, collapse = ", ")
  )
}

xgb_matrix <- as.matrix(
  test_data[, xgb_metadata$predictors, drop = FALSE]
)

xgb_probability <- predict(
  xgb_model,
  xgb_matrix
)

ensemble_probability <- (
  elastic_probability +
  rf_probability +
  xgb_probability
) / 3

# ------------------------------------------------------------
# 5. Load the thresholds selected using cross-validation
# ------------------------------------------------------------

elastic_threshold <- read.csv(
  "results/elastic_net_cv_performance.csv"
)$threshold[1]

rf_threshold <- read.csv(
  "results/random_forest_cv_performance.csv"
)$threshold[1]

xgb_threshold <- read.csv(
  "results/xgboost_native_cv_performance.csv"
)$threshold[1]

ensemble_threshold <- read.csv(
  "results/ensemble_cv_performance.csv"
)$threshold[1]

# ------------------------------------------------------------
# 6. Store participant-level test predictions
# ------------------------------------------------------------

test_predictions <- data.frame(
  Study.ID = test_data$Study.ID,
  observed = test_data$PCRpositive,

  elastic_probability = elastic_probability,
  elastic_class = ifelse(
    elastic_probability >= elastic_threshold,
    "Positive",
    "Negative"
  ),

  rf_probability = rf_probability,
  rf_class = ifelse(
    rf_probability >= rf_threshold,
    "Positive",
    "Negative"
  ),

  xgb_probability = xgb_probability,
  xgb_class = ifelse(
    xgb_probability >= xgb_threshold,
    "Positive",
    "Negative"
  ),

  ensemble_probability = ensemble_probability,
  ensemble_class = ifelse(
    ensemble_probability >= ensemble_threshold,
    "Positive",
    "Negative"
  )
)

write.csv(
  test_predictions,
  "results/external_test_predictions.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. Function for final model evaluation
# ------------------------------------------------------------

evaluate_model <- function(
  model_name,
  probability,
  threshold,
  observed
) {

  predicted_class <- factor(
    ifelse(
      probability >= threshold,
      "Positive",
      "Negative"
    ),
    levels = c("Positive", "Negative")
  )

  observed <- factor(
    observed,
    levels = c("Positive", "Negative")
  )

  confusion <- confusionMatrix(
    data = predicted_class,
    reference = observed,
    positive = "Positive"
  )

  roc_object <- roc(
    response = observed,
    predictor = probability,
    levels = c("Negative", "Positive"),
    direction = "<",
    quiet = TRUE
  )

  auc_ci <- ci.auc(
    roc_object,
    method = "delong"
  )

  tp <- unname(confusion$table["Positive", "Positive"])
  fp <- unname(confusion$table["Positive", "Negative"])
  fn <- unname(confusion$table["Negative", "Positive"])
  tn <- unname(confusion$table["Negative", "Negative"])

  sensitivity_ci <- binom.test(
    tp,
    tp + fn
  )$conf.int

  specificity_ci <- binom.test(
    tn,
    tn + fp
  )$conf.int

  performance <- data.frame(
    model = model_name,
    threshold = threshold,
    ROC_AUC = as.numeric(auc(roc_object)),
    AUC_lower_95 = as.numeric(auc_ci[1]),
    AUC_upper_95 = as.numeric(auc_ci[3]),
    sensitivity = unname(
      confusion$byClass["Sensitivity"]
    ),
    sensitivity_lower_95 = sensitivity_ci[1],
    sensitivity_upper_95 = sensitivity_ci[2],
    specificity = unname(
      confusion$byClass["Specificity"]
    ),
    specificity_lower_95 = specificity_ci[1],
    specificity_upper_95 = specificity_ci[2],
    balanced_accuracy = unname(
      confusion$byClass["Balanced Accuracy"]
    ),
    accuracy = unname(
      confusion$overall["Accuracy"]
    ),
    TP = tp,
    FP = fp,
    FN = fn,
    TN = tn
  )

  list(
    performance = performance,
    confusion = confusion$table
  )
}

# ------------------------------------------------------------
# 8. Evaluate all models
# ------------------------------------------------------------

elastic_results <- evaluate_model(
  model_name = "Elastic Net",
  probability = elastic_probability,
  threshold = elastic_threshold,
  observed = test_data$PCRpositive
)

rf_results <- evaluate_model(
  model_name = "Random Forest",
  probability = rf_probability,
  threshold = rf_threshold,
  observed = test_data$PCRpositive
)

xgb_results <- evaluate_model(
  model_name = "XGBoost",
  probability = xgb_probability,
  threshold = xgb_threshold,
  observed = test_data$PCRpositive
)

ensemble_results <- evaluate_model(
  model_name = "Equal-weight ensemble",
  probability = ensemble_probability,
  threshold = ensemble_threshold,
  observed = test_data$PCRpositive
)

test_performance <- bind_rows(
  elastic_results$performance,
  rf_results$performance,
  xgb_results$performance,
  ensemble_results$performance
) %>%
  arrange(desc(ROC_AUC))

write.csv(
  test_performance,
  "results/external_test_model_comparison.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 9. Print final results
# ------------------------------------------------------------

cat("\nFinal external test evaluation completed.\n")
cat("Test participants:", nrow(test_data), "\n")
cat(
  "Positive:",
  sum(test_data$PCRpositive == "Positive"),
  "\n"
)
cat(
  "Negative:",
  sum(test_data$PCRpositive == "Negative"),
  "\n\n"
)

cat("External test performance:\n")
print(test_performance)

cat("\nElastic Net confusion matrix:\n")
print(elastic_results$confusion)

cat("\nRandom Forest confusion matrix:\n")
print(rf_results$confusion)

cat("\nXGBoost confusion matrix:\n")
print(xgb_results$confusion)

cat("\nEqual-weight ensemble confusion matrix:\n")
print(ensemble_results$confusion)

cat(
  "\nThis was the final evaluation. ",
  "Do not tune or select models using these test results.\n",
  sep = ""
)
