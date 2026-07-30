# R/16_vaccination_sensitivity_cv.R
# Post-hoc sensitivity analysis:
# compare the original 16 immune predictors with
# 16 immune predictors + vaccination status.
#
# This script uses TRAINING DATA ONLY.
# It uses the same repeated 5-fold CV partitions and the same
# fitted-model settings selected in the primary analysis.
# The held-out test set is not used.

library(glmnet)
library(ranger)
library(xgboost)
library(dplyr)
library(pROC)

# ------------------------------------------------------------
# 1. Files
# ------------------------------------------------------------

training_file <- paste0(
  "data/processed/",
  "training_data_16_plus_vaccination.csv"
)

folds_file <- "models/cross_validation_folds.rds"

dir.create("results", showWarnings = FALSE)

# ------------------------------------------------------------
# 2. Read training data
# ------------------------------------------------------------

training_data <- read.csv(
  training_file,
  stringsAsFactors = FALSE,
  check.names = TRUE
)

training_data$PCRpositive <- factor(
  training_data$PCRpositive,
  levels = c("Positive", "Negative")
)

training_data$Flu_Vaccine_2018 <- factor(
  training_data$Flu_Vaccine_2018,
  levels = c("Unvaccinated", "Vaccinated")
)

if (any(is.na(training_data$Flu_Vaccine_2018))) {
  stop("Vaccination status contains missing or unexpected values.")
}

predictor_names <- setdiff(
  names(training_data),
  c("Study.ID", "PCRpositive")
)

# Numeric matrix for Elastic Net and XGBoost.
# Unvaccinated is the reference level, so this creates one
# vaccination indicator: Flu_Vaccine_2018Vaccinated.
x_matrix <- model.matrix(
  ~ .,
  data = training_data[, predictor_names, drop = FALSE]
)[, -1, drop = FALSE]

# Symptomatic influenza = 1; protected = 0.
y_numeric <- ifelse(
  training_data$PCRpositive == "Positive",
  1,
  0
)

# Data frame for Random Forest.
rf_data <- training_data %>%
  select(-Study.ID)

# ------------------------------------------------------------
# 3. Load the exact cross-validation folds
# ------------------------------------------------------------

cv_folds <- readRDS(folds_file)

all_rows <- seq_len(nrow(training_data))

if (nrow(training_data) != 134) {
  stop(
    "Expected 134 training participants but found ",
    nrow(training_data),
    "."
  )
}

# ------------------------------------------------------------
# 4. Fixed settings from the primary analysis
# ------------------------------------------------------------

elastic_alpha <- 0
elastic_lambda <- 0.385662

rf_mtry <- 2
rf_min_node_size <- 5
rf_num_trees <- 1000

xgb_nrounds <- 100
xgb_eta <- 0.10
xgb_max_depth <- 1
xgb_min_child_weight <- 1
xgb_subsample <- 0.8
xgb_colsample_bytree <- 0.8
xgb_gamma <- 0

# ------------------------------------------------------------
# 5. Generate repeated out-of-fold predictions
# ------------------------------------------------------------

set.seed(2026)

oof_list <- vector(
  "list",
  length(cv_folds)
)

for (fold_number in seq_along(cv_folds)) {

  train_index <- cv_folds[[fold_number]]
  validation_index <- setdiff(
    all_rows,
    train_index
  )

  # Elastic Net
  elastic_model <- glmnet(
    x = x_matrix[train_index, , drop = FALSE],
    y = y_numeric[train_index],
    family = "binomial",
    alpha = elastic_alpha,
    lambda = elastic_lambda,
    standardize = TRUE
  )

  elastic_probability <- as.numeric(
    predict(
      elastic_model,
      newx = x_matrix[
        validation_index,
        ,
        drop = FALSE
      ],
      s = elastic_lambda,
      type = "response"
    )
  )

  # Random Forest
  rf_model <- ranger(
    PCRpositive ~ .,
    data = rf_data[
      train_index,
      ,
      drop = FALSE
    ],
    probability = TRUE,
    num.trees = rf_num_trees,
    mtry = rf_mtry,
    splitrule = "gini",
    min.node.size = rf_min_node_size,
    respect.unordered.factors = "order",
    num.threads = 2,
    seed = 2026 + fold_number
  )

  rf_prediction_matrix <- predict(
    rf_model,
    data = rf_data[
      validation_index,
      predictor_names,
      drop = FALSE
    ]
  )$predictions

  if (!"Positive" %in% colnames(rf_prediction_matrix)) {
    stop(
      "Random Forest did not return a Positive probability column."
    )
  }

  rf_probability <- as.numeric(
    rf_prediction_matrix[, "Positive"]
  )

  # XGBoost
  xgb_training_matrix <- xgb.DMatrix(
    data = x_matrix[
      train_index,
      ,
      drop = FALSE
    ],
    label = y_numeric[train_index]
  )

  xgb_model <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = xgb_eta,
      max_depth = xgb_max_depth,
      min_child_weight = xgb_min_child_weight,
      subsample = xgb_subsample,
      colsample_bytree = xgb_colsample_bytree,
      gamma = xgb_gamma,
      nthread = 2
    ),
    data = xgb_training_matrix,
    nrounds = xgb_nrounds,
    verbose = 0
  )

  xgb_probability <- as.numeric(
    predict(
      xgb_model,
      x_matrix[
        validation_index,
        ,
        drop = FALSE
      ]
    )
  )

  ensemble_probability <- (
    elastic_probability +
    rf_probability +
    xgb_probability
  ) / 3

  oof_list[[fold_number]] <- data.frame(
    rowIndex = validation_index,
    observed = y_numeric[validation_index],
    elastic_probability = elastic_probability,
    rf_probability = rf_probability,
    xgb_probability = xgb_probability,
    ensemble_probability = ensemble_probability
  )

  cat(
    "Completed fold",
    fold_number,
    "of",
    length(cv_folds),
    "\n"
  )
}

# ------------------------------------------------------------
# 6. Average repeated out-of-fold predictions
# ------------------------------------------------------------

oof_predictions <- bind_rows(oof_list) %>%
  group_by(rowIndex, observed) %>%
  summarise(
    elastic_probability =
      mean(elastic_probability),
    rf_probability =
      mean(rf_probability),
    xgb_probability =
      mean(xgb_probability),
    ensemble_probability =
      mean(ensemble_probability),
    .groups = "drop"
  )

if (nrow(oof_predictions) != 134) {
  stop(
    "Expected one averaged prediction for 134 participants."
  )
}

write.csv(
  oof_predictions,
  paste0(
    "results/",
    "vaccination_sensitivity_cv_predictions.csv"
  ),
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. Evaluate each model using Youden's index
# ------------------------------------------------------------

evaluate_oof_model <- function(
  model_name,
  probability,
  observed
) {

  roc_object <- roc(
    response = observed,
    predictor = probability,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )

  best_threshold <- coords(
    roc_object,
    x = "best",
    best.method = "youden",
    ret = c(
      "threshold",
      "sensitivity",
      "specificity"
    ),
    transpose = FALSE
  )

  threshold <- as.numeric(
    best_threshold$threshold
  )

  predicted <- ifelse(
    probability >= threshold,
    1,
    0
  )

  true_positive <- sum(
    predicted == 1 & observed == 1
  )

  false_positive <- sum(
    predicted == 1 & observed == 0
  )

  false_negative <- sum(
    predicted == 0 & observed == 1
  )

  true_negative <- sum(
    predicted == 0 & observed == 0
  )

  sensitivity <- true_positive / (
    true_positive + false_negative
  )

  specificity <- true_negative / (
    true_negative + false_positive
  )

  data.frame(
    model = model_name,
    threshold = threshold,
    ROC_AUC = as.numeric(auc(roc_object)),
    sensitivity = sensitivity,
    specificity = specificity,
    balanced_accuracy = mean(
      c(sensitivity, specificity)
    ),
    accuracy = mean(
      predicted == observed
    ),
    TP = true_positive,
    FP = false_positive,
    FN = false_negative,
    TN = true_negative
  )
}

sensitivity_performance <- bind_rows(
  evaluate_oof_model(
    "Elastic Net",
    oof_predictions$elastic_probability,
    oof_predictions$observed
  ),
  evaluate_oof_model(
    "Random Forest",
    oof_predictions$rf_probability,
    oof_predictions$observed
  ),
  evaluate_oof_model(
    "XGBoost",
    oof_predictions$xgb_probability,
    oof_predictions$observed
  ),
  evaluate_oof_model(
    "Equal-weight ensemble",
    oof_predictions$ensemble_probability,
    oof_predictions$observed
  )
) %>%
  mutate(
    analysis =
      "Post-hoc: 16 immune predictors + vaccination"
  ) %>%
  select(
    analysis,
    everything()
  ) %>%
  arrange(desc(ROC_AUC))

write.csv(
  sensitivity_performance,
  paste0(
    "results/",
    "vaccination_sensitivity_cv_performance.csv"
  ),
  row.names = FALSE
)

# ------------------------------------------------------------
# 8. Compare with the primary cross-validation results
# ------------------------------------------------------------

primary_performance <- read.csv(
  "results/cv_model_comparison.csv",
  stringsAsFactors = FALSE
) %>%
  mutate(
    analysis =
      "Primary: 16 immune predictors"
  ) %>%
  select(
    analysis,
    model,
    threshold,
    ROC_AUC,
    sensitivity,
    specificity,
    balanced_accuracy,
    accuracy
  )

combined_comparison <- bind_rows(
  primary_performance,
  sensitivity_performance %>%
    select(
      analysis,
      model,
      threshold,
      ROC_AUC,
      sensitivity,
      specificity,
      balanced_accuracy,
      accuracy
    )
) %>%
  arrange(
    model,
    analysis
  )

write.csv(
  combined_comparison,
  paste0(
    "results/",
    "primary_vs_vaccination_sensitivity_cv.csv"
  ),
  row.names = FALSE
)

# ------------------------------------------------------------
# 9. Fit full training Elastic Net to inspect vaccination effect
# ------------------------------------------------------------

full_elastic_model <- glmnet(
  x = x_matrix,
  y = y_numeric,
  family = "binomial",
  alpha = elastic_alpha,
  lambda = elastic_lambda,
  standardize = TRUE
)

elastic_coefficients <- as.matrix(
  coef(
    full_elastic_model,
    s = elastic_lambda
  )
)

vaccination_row <- grep(
  "Flu_Vaccine_2018",
  rownames(elastic_coefficients),
  value = TRUE
)

vaccination_effect <- if (
  length(vaccination_row) == 1
) {
  coefficient <- as.numeric(
    elastic_coefficients[
      vaccination_row,
      1
    ]
  )

  data.frame(
    variable = vaccination_row,
    coefficient = coefficient,
    odds_ratio = exp(coefficient)
  )
} else {
  data.frame(
    variable = NA_character_,
    coefficient = NA_real_,
    odds_ratio = NA_real_
  )
}

write.csv(
  vaccination_effect,
  paste0(
    "results/",
    "vaccination_elastic_net_effect.csv"
  ),
  row.names = FALSE
)

# ------------------------------------------------------------
# 10. Print results
# ------------------------------------------------------------

cat(
  "\nVaccination sensitivity cross-validation completed.\n\n"
)

cat(
  "Post-hoc performance with vaccination added:\n"
)
print(sensitivity_performance)

cat(
  "\nPrimary versus vaccination-adjusted comparison:\n"
)
print(combined_comparison)

cat(
  "\nElastic Net vaccination coefficient:\n"
)
print(vaccination_effect)

cat(
  "\nThe held-out test set was not used in this script.\n"
)
