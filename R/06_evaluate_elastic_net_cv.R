# R/06_evaluate_elastic_net_cv.R
# Evaluate Elastic Net using cross-validated training predictions only.
# The external test set remains untouched.

library(caret)
library(dplyr)
library(pROC)
library(ggplot2)

# ------------------------------------------------------------
# 1. File locations
# ------------------------------------------------------------

model_file <- "models/elastic_net_model.rds"
result_dir <- "results"

dir.create(result_dir, showWarnings = FALSE)

# ------------------------------------------------------------
# 2. Load the trained Elastic Net model
# ------------------------------------------------------------

elastic_net_model <- readRDS(model_file)

# caret saved out-of-fold predictions for the best alpha/lambda values.
cv_predictions <- elastic_net_model$pred

# ------------------------------------------------------------
# 3. Confirm that Positive is the event being predicted
# ------------------------------------------------------------

required_columns <- c("rowIndex", "obs", "Positive")

missing_columns <- setdiff(required_columns, names(cv_predictions))

if (length(missing_columns) > 0) {
  stop(
    "Required prediction columns are missing: ",
    paste(missing_columns, collapse = ", ")
  )
}

# Each participant was predicted once per repeat.
# Average those repeated out-of-fold probabilities to obtain
# one cross-validated probability per training participant.
cv_averaged <- cv_predictions %>%
  group_by(rowIndex, obs) %>%
  summarise(
    predicted_probability = mean(Positive),
    .groups = "drop"
  )

cv_averaged$obs <- factor(
  cv_averaged$obs,
  levels = c("Positive", "Negative")
)

# ------------------------------------------------------------
# 4. Create the ROC curve
# ------------------------------------------------------------

# Negative is the control class and Positive is the case class.
# direction = "<" means higher probabilities indicate Positive cases.
roc_object <- roc(
  response = cv_averaged$obs,
  predictor = cv_averaged$predicted_probability,
  levels = c("Negative", "Positive"),
  direction = "<",
  quiet = TRUE
)

cv_auc <- as.numeric(auc(roc_object))

# ------------------------------------------------------------
# 5. Select a classification threshold
# ------------------------------------------------------------

# Youden's index chooses the threshold that maximises:
# sensitivity + specificity - 1
# This gives equal importance to detecting cases and controls.
best_threshold <- coords(
  roc_object,
  x = "best",
  best.method = "youden",
  ret = c("threshold", "sensitivity", "specificity"),
  transpose = FALSE
)

threshold_value <- as.numeric(best_threshold$threshold)

# Convert probabilities into Positive/Negative predictions
# using the threshold selected from training data only.
cv_averaged <- cv_averaged %>%
  mutate(
    predicted_class = factor(
      ifelse(
        predicted_probability >= threshold_value,
        "Positive",
        "Negative"
      ),
      levels = c("Positive", "Negative")
    )
  )

# ------------------------------------------------------------
# 6. Calculate classification performance
# ------------------------------------------------------------

confusion_results <- confusionMatrix(
  data = cv_averaged$predicted_class,
  reference = cv_averaged$obs,
  positive = "Positive"
)

performance_table <- data.frame(
  threshold = threshold_value,
  ROC_AUC = cv_auc,
  sensitivity = unname(confusion_results$byClass["Sensitivity"]),
  specificity = unname(confusion_results$byClass["Specificity"]),
  balanced_accuracy = unname(
    confusion_results$byClass["Balanced Accuracy"]
  ),
  accuracy = unname(confusion_results$overall["Accuracy"])
)

# ------------------------------------------------------------
# 7. Save predictions and results
# ------------------------------------------------------------

write.csv(
  cv_averaged,
  file.path(result_dir, "elastic_net_cv_predictions.csv"),
  row.names = FALSE
)

write.csv(
  performance_table,
  file.path(result_dir, "elastic_net_cv_performance.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 8. Save the ROC curve
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
    title = "Elastic Net: Cross-validated ROC",
    subtitle = paste0("AUC = ", round(cv_auc, 3)),
    x = "False positive rate",
    y = "True positive rate"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(result_dir, "elastic_net_cv_roc.png"),
  plot = roc_plot,
  width = 6,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 9. Print the main findings
# ------------------------------------------------------------

cat("\nElastic Net cross-validation evaluation completed.\n\n")

cat("Higher predicted probability represents symptomatic influenza.\n\n")

cat("Selected threshold:\n")
print(round(threshold_value, 4))

cat("\nCross-validated performance at this threshold:\n")
print(performance_table)

cat("\nConfusion matrix:\n")
print(confusion_results$table)

cat("\nThe external test set has not been used.\n")
