# R/10_build_cv_ensemble.R
# Build an equal-weight ensemble from the cross-validated predictions of:
#   1. Elastic Net
#   2. Random Forest
#   3. XGBoost
#
# Only out-of-fold TRAINING predictions are used.
# The external test set remains untouched.

library(dplyr)
library(pROC)
library(caret)
library(ggplot2)

# ------------------------------------------------------------
# 1. File locations
# ------------------------------------------------------------

elastic_file <- "results/elastic_net_cv_predictions.csv"
rf_file <- "results/random_forest_cv_predictions.csv"
xgb_file <- "results/xgboost_native_cv_predictions.csv"

result_dir <- "results"
dir.create(result_dir, showWarnings = FALSE)

# ------------------------------------------------------------
# 2. Read each model's cross-validated predictions
# ------------------------------------------------------------

elastic <- read.csv(
  elastic_file,
  stringsAsFactors = FALSE
) %>%
  select(
    rowIndex,
    elastic_observed = obs,
    elastic_probability = predicted_probability
  )

random_forest <- read.csv(
  rf_file,
  stringsAsFactors = FALSE
) %>%
  select(
    rowIndex,
    rf_observed = obs,
    rf_probability = predicted_probability
  )

xgboost <- read.csv(
  xgb_file,
  stringsAsFactors = FALSE
) %>%
  select(
    rowIndex,
    xgb_observed = observed,
    xgb_probability = predicted_probability
  )

# ------------------------------------------------------------
# 3. Join predictions by participant row
# ------------------------------------------------------------

ensemble_data <- elastic %>%
  inner_join(random_forest, by = "rowIndex") %>%
  inner_join(xgboost, by = "rowIndex")

# ------------------------------------------------------------
# 4. Confirm outcomes agree across all models
# ------------------------------------------------------------

# Convert XGBoost's numeric outcome to the same labels.
ensemble_data <- ensemble_data %>%
  mutate(
    xgb_observed_label = ifelse(
      xgb_observed == 1,
      "Positive",
      "Negative"
    )
  )

if (
  any(ensemble_data$elastic_observed != ensemble_data$rf_observed) ||
  any(ensemble_data$elastic_observed != ensemble_data$xgb_observed_label)
) {
  stop("Outcome labels do not agree across model prediction files.")
}

# Keep one outcome column.
ensemble_data <- ensemble_data %>%
  mutate(
    observed = factor(
      elastic_observed,
      levels = c("Positive", "Negative")
    )
  )

# ------------------------------------------------------------
# 5. Calculate equal-weight ensemble probability
# ------------------------------------------------------------

# Each model contributes one third of the final probability.
ensemble_data <- ensemble_data %>%
  mutate(
    ensemble_probability = (
      elastic_probability +
      rf_probability +
      xgb_probability
    ) / 3
  )

# ------------------------------------------------------------
# 6. Calculate ensemble ROC AUC
# ------------------------------------------------------------

roc_object <- roc(
  response = ensemble_data$observed,
  predictor = ensemble_data$ensemble_probability,
  levels = c("Negative", "Positive"),
  direction = "<",
  quiet = TRUE
)

ensemble_auc <- as.numeric(auc(roc_object))

# ------------------------------------------------------------
# 7. Select a threshold from training predictions only
# ------------------------------------------------------------

best_threshold <- coords(
  roc_object,
  x = "best",
  best.method = "youden",
  ret = c("threshold", "sensitivity", "specificity"),
  transpose = FALSE
)

threshold_value <- as.numeric(best_threshold$threshold)

ensemble_data <- ensemble_data %>%
  mutate(
    ensemble_class = factor(
      ifelse(
        ensemble_probability >= threshold_value,
        "Positive",
        "Negative"
      ),
      levels = c("Positive", "Negative")
    )
  )

# ------------------------------------------------------------
# 8. Calculate classification performance
# ------------------------------------------------------------

confusion_results <- confusionMatrix(
  data = ensemble_data$ensemble_class,
  reference = ensemble_data$observed,
  positive = "Positive"
)

ensemble_performance <- data.frame(
  model = "Equal-weight ensemble",
  threshold = threshold_value,
  ROC_AUC = ensemble_auc,
  sensitivity = unname(
    confusion_results$byClass["Sensitivity"]
  ),
  specificity = unname(
    confusion_results$byClass["Specificity"]
  ),
  balanced_accuracy = unname(
    confusion_results$byClass["Balanced Accuracy"]
  ),
  accuracy = unname(
    confusion_results$overall["Accuracy"]
  )
)

# ------------------------------------------------------------
# 9. Create a comparison table for all four models
# ------------------------------------------------------------

elastic_performance <- read.csv(
  "results/elastic_net_cv_performance.csv"
) %>%
  mutate(model = "Elastic Net")

rf_performance <- read.csv(
  "results/random_forest_cv_performance.csv"
) %>%
  mutate(model = "Random Forest")

xgb_performance <- read.csv(
  "results/xgboost_native_cv_performance.csv"
) %>%
  mutate(model = "XGBoost")

model_comparison <- bind_rows(
  elastic_performance,
  rf_performance,
  xgb_performance,
  ensemble_performance
) %>%
  select(
    model,
    threshold,
    ROC_AUC,
    sensitivity,
    specificity,
    balanced_accuracy,
    accuracy
  ) %>%
  arrange(desc(ROC_AUC))

# ------------------------------------------------------------
# 10. Save ensemble predictions and results
# ------------------------------------------------------------

write.csv(
  ensemble_data,
  file.path(result_dir, "ensemble_cv_predictions.csv"),
  row.names = FALSE
)

write.csv(
  ensemble_performance,
  file.path(result_dir, "ensemble_cv_performance.csv"),
  row.names = FALSE
)

write.csv(
  model_comparison,
  file.path(result_dir, "cv_model_comparison.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 11. Save the ensemble ROC plot
# ------------------------------------------------------------

roc_data <- data.frame(
  specificity = roc_object$specificities,
  sensitivity = roc_object$sensitivities
)

roc_plot <- ggplot(
  roc_data,
  aes(x = 1 - specificity, y = sensitivity)
) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = 2) +
  coord_equal() +
  labs(
    title = "Equal-weight ensemble: Cross-validated ROC",
    subtitle = paste0("AUC = ", round(ensemble_auc, 3)),
    x = "False positive rate",
    y = "True positive rate"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(
    result_dir,
    "ensemble_cv_roc.png"
  ),
  plot = roc_plot,
  width = 6,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 12. Print main results
# ------------------------------------------------------------

cat("\nEqual-weight ensemble completed.\n\n")

cat("Selected threshold:\n")
print(round(threshold_value, 4))

cat("\nCross-validated ensemble performance:\n")
print(ensemble_performance)

cat("\nConfusion matrix:\n")
print(confusion_results$table)

cat("\nComparison of all models:\n")
print(model_comparison)

cat("\nThe external test set has not been used.\n")
