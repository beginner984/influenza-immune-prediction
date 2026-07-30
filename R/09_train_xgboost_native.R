# R/09_train_xgboost_native.R
# Train XGBoost directly with xgboost 3.x.
# Uses the same training data and saved cross-validation folds.
# The external test set remains untouched.

library(xgboost)
library(dplyr)
library(pROC)

training_file <- "data/processed/training_data_16.csv"
folds_file <- "models/cross_validation_folds.rds"

dir.create("models", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

training_data <- read.csv(
  training_file,
  stringsAsFactors = FALSE,
  check.names = TRUE
)

selected_predictors <- setdiff(
  names(training_data),
  c("Study.ID", "PCRpositive")
)

x <- as.matrix(training_data[, selected_predictors])
y <- ifelse(training_data$PCRpositive == "Positive", 1, 0)

cv_folds <- readRDS(folds_file)
all_rows <- seq_len(nrow(training_data))

# A moderate grid suitable for this small dataset.
tuning_grid <- expand.grid(
  nrounds = c(100, 300),
  eta = c(0.03, 0.10),
  max_depth = c(1, 2, 3),
  min_child_weight = c(1, 5),
  KEEP.OUT.ATTRS = FALSE
)

fixed_subsample <- 0.8
fixed_colsample <- 0.8
fixed_gamma <- 0

set.seed(2026)
tuning_results <- vector("list", nrow(tuning_grid))

for (g in seq_len(nrow(tuning_grid))) {

  fold_auc <- numeric(length(cv_folds))

  for (f in seq_along(cv_folds)) {

    train_index <- cv_folds[[f]]
    validation_index <- setdiff(all_rows, train_index)

    dtrain <- xgb.DMatrix(
      data = x[train_index, , drop = FALSE],
      label = y[train_index]
    )

    model <- xgb.train(
      params = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        eta = tuning_grid$eta[g],
        max_depth = tuning_grid$max_depth[g],
        min_child_weight = tuning_grid$min_child_weight[g],
        subsample = fixed_subsample,
        colsample_bytree = fixed_colsample,
        gamma = fixed_gamma,
        nthread = 2
      ),
      data = dtrain,
      nrounds = tuning_grid$nrounds[g],
      verbose = 0
    )

    validation_prob <- predict(
      model,
      x[validation_index, , drop = FALSE]
    )

    fold_auc[f] <- as.numeric(
      auc(
        roc(
          response = y[validation_index],
          predictor = validation_prob,
          levels = c(0, 1),
          direction = "<",
          quiet = TRUE
        )
      )
    )
  }

  tuning_results[[g]] <- data.frame(
    tuning_grid[g, ],
    mean_ROC = mean(fold_auc),
    sd_ROC = sd(fold_auc)
  )

  cat(
    "Completed", g, "of", nrow(tuning_grid),
    "- mean AUC:", round(mean(fold_auc), 3), "\n"
  )
}

tuning_results <- bind_rows(tuning_results) %>%
  arrange(desc(mean_ROC))

best_params <- tuning_results[1, ]

write.csv(
  tuning_results,
  "results/xgboost_native_tuning_results.csv",
  row.names = FALSE
)

write.csv(
  best_params,
  "results/xgboost_native_best_tuning.csv",
  row.names = FALSE
)

# Generate repeated out-of-fold probabilities with the best settings.
oof_long <- vector("list", length(cv_folds))

for (f in seq_along(cv_folds)) {

  train_index <- cv_folds[[f]]
  validation_index <- setdiff(all_rows, train_index)

  dtrain <- xgb.DMatrix(
    data = x[train_index, , drop = FALSE],
    label = y[train_index]
  )

  model <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = best_params$eta,
      max_depth = best_params$max_depth,
      min_child_weight = best_params$min_child_weight,
      subsample = fixed_subsample,
      colsample_bytree = fixed_colsample,
      gamma = fixed_gamma,
      nthread = 2
    ),
    data = dtrain,
    nrounds = best_params$nrounds,
    verbose = 0
  )

  oof_long[[f]] <- data.frame(
    rowIndex = validation_index,
    observed = y[validation_index],
    predicted_probability = predict(
      model,
      x[validation_index, , drop = FALSE]
    )
  )
}

oof_predictions <- bind_rows(oof_long) %>%
  group_by(rowIndex, observed) %>%
  summarise(
    predicted_probability = mean(predicted_probability),
    .groups = "drop"
  )

roc_object <- roc(
  response = oof_predictions$observed,
  predictor = oof_predictions$predicted_probability,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

best_threshold <- coords(
  roc_object,
  x = "best",
  best.method = "youden",
  ret = c("threshold", "sensitivity", "specificity"),
  transpose = FALSE
)

threshold_value <- as.numeric(best_threshold$threshold)

oof_predictions <- oof_predictions %>%
  mutate(
    predicted_class = ifelse(
      predicted_probability >= threshold_value,
      1,
      0
    )
  )

true_positive <- sum(
  oof_predictions$predicted_class == 1 &
    oof_predictions$observed == 1
)

false_negative <- sum(
  oof_predictions$predicted_class == 0 &
    oof_predictions$observed == 1
)

true_negative <- sum(
  oof_predictions$predicted_class == 0 &
    oof_predictions$observed == 0
)

false_positive <- sum(
  oof_predictions$predicted_class == 1 &
    oof_predictions$observed == 0
)

sensitivity <- true_positive / (true_positive + false_negative)
specificity <- true_negative / (true_negative + false_positive)

performance <- data.frame(
  threshold = threshold_value,
  ROC_AUC = as.numeric(auc(roc_object)),
  sensitivity = sensitivity,
  specificity = specificity,
  balanced_accuracy = mean(c(sensitivity, specificity)),
  accuracy = mean(
    oof_predictions$predicted_class == oof_predictions$observed
  )
)

write.csv(
  oof_predictions,
  "results/xgboost_native_cv_predictions.csv",
  row.names = FALSE
)

write.csv(
  performance,
  "results/xgboost_native_cv_performance.csv",
  row.names = FALSE
)

# Fit the final model on all training participants.
full_dtrain <- xgb.DMatrix(data = x, label = y)

final_xgboost_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = best_params$eta,
    max_depth = best_params$max_depth,
    min_child_weight = best_params$min_child_weight,
    subsample = fixed_subsample,
    colsample_bytree = fixed_colsample,
    gamma = fixed_gamma,
    nthread = 2
  ),
  data = full_dtrain,
  nrounds = best_params$nrounds,
  verbose = 0
)

xgb.save(
  final_xgboost_model,
  "models/xgboost_native_model.json"
)

saveRDS(
  list(
    predictors = selected_predictors,
    best_params = best_params,
    threshold = threshold_value
  ),
  "models/xgboost_native_metadata.rds"
)

importance_table <- xgb.importance(
  feature_names = selected_predictors,
  model = final_xgboost_model
)

write.csv(
  importance_table,
  "results/xgboost_native_variable_importance.csv",
  row.names = FALSE
)

cat("\nNative XGBoost training completed.\n\n")
cat("Best tuning values:\n")
print(best_params)

cat("\nCross-validated performance:\n")
print(performance)

cat("\nConfusion matrix:\n")
print(
  matrix(
    c(true_positive, false_positive, false_negative, true_negative),
    nrow = 2,
    byrow = TRUE,
    dimnames = list(
      Prediction = c("Positive", "Negative"),
      Reference = c("Positive", "Negative")
    )
  )
)

cat("\nTop 10 variables:\n")
print(head(importance_table, 10))

cat("\nThe external test set has not been used.\n")
