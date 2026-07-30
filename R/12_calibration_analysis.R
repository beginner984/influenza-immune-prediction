# R/12_calibration_analysis.R
# Calibration analysis for Elastic Net, Random Forest, XGBoost,
# and the equal-weight ensemble.
#
# Training calibration uses averaged out-of-fold predictions.
# Isotonic calibration of the final ensemble is cross-fitted on training data.
# A final isotonic mapping is then fitted using all training OOF predictions
# and applied once to the held-out test probabilities.

library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(pROC)

dir.create("results", showWarnings = FALSE)

# ------------------------------------------------------------
# 1. Helper functions
# ------------------------------------------------------------

clip_probability <- function(p, epsilon = 1e-6) {
  pmin(pmax(as.numeric(p), epsilon), 1 - epsilon)
}

brier_score <- function(y, p) {
  mean((as.numeric(p) - as.numeric(y))^2)
}

calibration_intercept_slope <- function(y, p) {
  p <- clip_probability(p)
  lp <- qlogis(p)

  intercept_fit <- glm(
    y ~ 1,
    offset = lp,
    family = binomial()
  )

  slope_fit <- glm(
    y ~ lp,
    family = binomial()
  )

  c(
    calibration_intercept = unname(coef(intercept_fit)[1]),
    calibration_slope = unname(coef(slope_fit)["lp"])
  )
}

hosmer_lemeshow <- function(y, p, groups = 10) {
  y <- as.numeric(y)
  p <- clip_probability(p)

  groups <- min(groups, floor(length(y) / 5))
  groups <- max(groups, 3)

  grouped <- data.frame(y = y, p = p) %>%
    arrange(p) %>%
    mutate(group = ntile(row_number(), groups)) %>%
    group_by(group) %>%
    summarise(
      n = n(),
      observed_positive = sum(y),
      expected_positive = sum(p),
      .groups = "drop"
    ) %>%
    mutate(
      observed_negative = n - observed_positive,
      expected_negative = n - expected_positive
    )

  chi_square <- sum(
    (grouped$observed_positive - grouped$expected_positive)^2 /
      pmax(grouped$expected_positive, 1e-8) +
    (grouped$observed_negative - grouped$expected_negative)^2 /
      pmax(grouped$expected_negative, 1e-8)
  )

  df <- nrow(grouped) - 2
  p_value <- pchisq(chi_square, df = df, lower.tail = FALSE)

  list(
    chi_square = chi_square,
    df = df,
    p_value = p_value,
    table = grouped
  )
}

calibration_metrics <- function(model, y, p, hl_groups = 10) {
  calibration_values <- calibration_intercept_slope(y, p)
  hl <- hosmer_lemeshow(y, p, groups = hl_groups)

  data.frame(
    model = model,
    Brier_score = brier_score(y, p),
    calibration_intercept =
      calibration_values["calibration_intercept"],
    calibration_slope =
      calibration_values["calibration_slope"],
    HL_chi_square = hl$chi_square,
    HL_df = hl$df,
    HL_p_value = hl$p_value
  )
}

make_calibration_bins <- function(data, probability_column, groups = 10) {
  probability_name <- rlang::ensym(probability_column)

  data %>%
    arrange(!!probability_name) %>%
    mutate(bin = ntile(row_number(), groups)) %>%
    group_by(bin) %>%
    summarise(
      mean_predicted = mean(!!probability_name),
      observed_fraction = mean(observed),
      n = n(),
      .groups = "drop"
    )
}

fit_isotonic_map <- function(probability, observed) {
  ordering <- order(probability)

  x_sorted <- as.numeric(probability[ordering])
  y_sorted <- as.numeric(observed[ordering])

  fit <- isoreg(x_sorted, y_sorted)

  mapping <- data.frame(
    x = x_sorted,
    calibrated = fit$yf
  ) %>%
    group_by(x) %>%
    summarise(
      calibrated = mean(calibrated),
      .groups = "drop"
    ) %>%
    arrange(x)

  list(
    x = mapping$x,
    calibrated = mapping$calibrated
  )
}

predict_isotonic <- function(mapping, probability) {
  prediction <- approx(
    x = mapping$x,
    y = mapping$calibrated,
    xout = as.numeric(probability),
    method = "constant",
    f = 1,
    rule = 2,
    ties = "ordered"
  )$y

  pmin(pmax(prediction, 0), 1)
}

crossfit_isotonic <- function(probability, observed, folds = 5, seed = 2026) {
  set.seed(seed)

  fold_indices <- createFolds(
    factor(observed, levels = c(0, 1)),
    k = folds,
    returnTrain = FALSE
  )

  calibrated <- rep(NA_real_, length(observed))

  for (i in seq_along(fold_indices)) {
    validation_index <- fold_indices[[i]]
    training_index <- setdiff(seq_along(observed), validation_index)

    mapping <- fit_isotonic_map(
      probability[training_index],
      observed[training_index]
    )

    calibrated[validation_index] <- predict_isotonic(
      mapping,
      probability[validation_index]
    )
  }

  calibrated
}

# ------------------------------------------------------------
# 2. Read cross-validated training predictions
# ------------------------------------------------------------

elastic <- read.csv(
  "results/elastic_net_cv_predictions.csv",
  stringsAsFactors = FALSE
) %>%
  select(
    rowIndex,
    observed_label = obs,
    Elastic_Net = predicted_probability
  )

random_forest <- read.csv(
  "results/random_forest_cv_predictions.csv",
  stringsAsFactors = FALSE
) %>%
  select(
    rowIndex,
    rf_observed = obs,
    Random_Forest = predicted_probability
  )

xgboost <- read.csv(
  "results/xgboost_native_cv_predictions.csv",
  stringsAsFactors = FALSE
) %>%
  select(
    rowIndex,
    xgb_observed = observed,
    XGBoost = predicted_probability
  )

ensemble <- read.csv(
  "results/ensemble_cv_predictions.csv",
  stringsAsFactors = FALSE
) %>%
  select(
    rowIndex,
    Equal_weight_ensemble = ensemble_probability
  )

calibration_data <- elastic %>%
  inner_join(random_forest, by = "rowIndex") %>%
  inner_join(xgboost, by = "rowIndex") %>%
  inner_join(ensemble, by = "rowIndex") %>%
  mutate(
    observed = ifelse(observed_label == "Positive", 1, 0),
    xgb_observed_label = ifelse(
      xgb_observed == 1,
      "Positive",
      "Negative"
    )
  )

if (
  any(calibration_data$observed_label != calibration_data$rf_observed) ||
  any(calibration_data$observed_label != calibration_data$xgb_observed_label)
) {
  stop("Outcome labels disagree across prediction files.")
}

# ------------------------------------------------------------
# 3. Calibration metrics for all models before scaling
# ------------------------------------------------------------

training_metrics <- bind_rows(
  calibration_metrics(
    "Elastic Net",
    calibration_data$observed,
    calibration_data$Elastic_Net
  ),
  calibration_metrics(
    "Random Forest",
    calibration_data$observed,
    calibration_data$Random_Forest
  ),
  calibration_metrics(
    "XGBoost",
    calibration_data$observed,
    calibration_data$XGBoost
  ),
  calibration_metrics(
    "Equal-weight ensemble",
    calibration_data$observed,
    calibration_data$Equal_weight_ensemble
  )
)

write.csv(
  training_metrics,
  "results/calibration_training_all_models.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 4. Cross-fitted isotonic calibration of final ensemble
# ------------------------------------------------------------

calibration_data$ensemble_isotonic_crossfit <- crossfit_isotonic(
  probability = calibration_data$Equal_weight_ensemble,
  observed = calibration_data$observed,
  folds = 5,
  seed = 2026
)

ensemble_training_comparison <- bind_rows(
  calibration_metrics(
    "Ensemble before isotonic scaling",
    calibration_data$observed,
    calibration_data$Equal_weight_ensemble
  ),
  calibration_metrics(
    "Ensemble after cross-fitted isotonic scaling",
    calibration_data$observed,
    calibration_data$ensemble_isotonic_crossfit
  )
)

write.csv(
  ensemble_training_comparison,
  "results/ensemble_calibration_training_before_after.csv",
  row.names = FALSE
)

write.csv(
  calibration_data,
  "results/calibration_training_predictions.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 5. Fit final isotonic mapping on all training OOF predictions
# ------------------------------------------------------------

final_isotonic_mapping <- fit_isotonic_map(
  calibration_data$Equal_weight_ensemble,
  calibration_data$observed
)

saveRDS(
  final_isotonic_mapping,
  "models/ensemble_isotonic_calibrator.rds"
)

# ------------------------------------------------------------
# 6. Apply calibration to held-out test probabilities
# ------------------------------------------------------------

test_predictions <- read.csv(
  "results/external_test_predictions.csv",
  stringsAsFactors = FALSE
)

test_predictions$observed_numeric <- ifelse(
  test_predictions$observed == "Positive",
  1,
  0
)

test_predictions$ensemble_probability_calibrated <- predict_isotonic(
  final_isotonic_mapping,
  test_predictions$ensemble_probability
)

test_metrics <- bind_rows(
  calibration_metrics(
    "Test ensemble before isotonic scaling",
    test_predictions$observed_numeric,
    test_predictions$ensemble_probability,
    hl_groups = 5
  ),
  calibration_metrics(
    "Test ensemble after isotonic scaling",
    test_predictions$observed_numeric,
    test_predictions$ensemble_probability_calibrated,
    hl_groups = 5
  )
)

write.csv(
  test_metrics,
  "results/ensemble_calibration_test_before_after.csv",
  row.names = FALSE
)

write.csv(
  test_predictions,
  "results/external_test_predictions_calibrated.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 7. Calibration plot for all models on training OOF predictions
# ------------------------------------------------------------

training_long <- calibration_data %>%
  select(
    observed,
    Elastic_Net,
    Random_Forest,
    XGBoost,
    Equal_weight_ensemble
  ) %>%
  pivot_longer(
    cols = -observed,
    names_to = "model",
    values_to = "probability"
  ) %>%
  mutate(
    model = recode(
      model,
      Elastic_Net = "Elastic Net",
      Random_Forest = "Random Forest",
      XGBoost = "XGBoost",
      Equal_weight_ensemble = "Equal-weight ensemble"
    )
  )

training_bins <- training_long %>%
  group_by(model) %>%
  arrange(probability, .by_group = TRUE) %>%
  mutate(bin = ntile(row_number(), 10)) %>%
  group_by(model, bin) %>%
  summarise(
    mean_predicted = mean(probability),
    observed_fraction = mean(observed),
    n = n(),
    .groups = "drop"
  )

all_models_plot <- ggplot(
  training_bins,
  aes(x = mean_predicted, y = observed_fraction)
) +
  geom_abline(linetype = 2) +
  geom_line() +
  geom_point(size = 2) +
  facet_wrap(~ model, ncol = 2) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Calibration using out-of-fold training predictions",
    x = "Mean predicted probability",
    y = "Observed symptomatic-influenza proportion"
  ) +
  theme_minimal()

ggsave(
  "results/calibration_all_models_training.png",
  all_models_plot,
  width = 9,
  height = 8,
  dpi = 300
)

# ------------------------------------------------------------
# 8. Ensemble before/after calibration plot on training data
# ------------------------------------------------------------

ensemble_training_long <- calibration_data %>%
  transmute(
    observed,
    Before = Equal_weight_ensemble,
    `After cross-fitted isotonic scaling` =
      ensemble_isotonic_crossfit
  ) %>%
  pivot_longer(
    cols = -observed,
    names_to = "calibration",
    values_to = "probability"
  )

ensemble_training_bins <- ensemble_training_long %>%
  group_by(calibration) %>%
  arrange(probability, .by_group = TRUE) %>%
  mutate(bin = ntile(row_number(), 10)) %>%
  group_by(calibration, bin) %>%
  summarise(
    mean_predicted = mean(probability),
    observed_fraction = mean(observed),
    n = n(),
    .groups = "drop"
  )

ensemble_training_plot <- ggplot(
  ensemble_training_bins,
  aes(
    x = mean_predicted,
    y = observed_fraction,
    linetype = calibration,
    shape = calibration
  )
) +
  geom_abline(linetype = 2) +
  geom_line() +
  geom_point(size = 2) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Ensemble calibration on training data",
    subtitle = "After-scaling values are cross-fitted",
    x = "Mean predicted probability",
    y = "Observed symptomatic-influenza proportion",
    linetype = NULL,
    shape = NULL
  ) +
  theme_minimal()

ggsave(
  "results/ensemble_calibration_training_before_after.png",
  ensemble_training_plot,
  width = 7,
  height = 6,
  dpi = 300
)

# ------------------------------------------------------------
# 9. Ensemble before/after calibration plot on held-out test data
# ------------------------------------------------------------

test_long <- test_predictions %>%
  transmute(
    observed = observed_numeric,
    Before = ensemble_probability,
    `After isotonic scaling` =
      ensemble_probability_calibrated
  ) %>%
  pivot_longer(
    cols = -observed,
    names_to = "calibration",
    values_to = "probability"
  )

test_bins <- test_long %>%
  group_by(calibration) %>%
  arrange(probability, .by_group = TRUE) %>%
  mutate(bin = ntile(row_number(), 5)) %>%
  group_by(calibration, bin) %>%
  summarise(
    mean_predicted = mean(probability),
    observed_fraction = mean(observed),
    n = n(),
    .groups = "drop"
  )

test_plot <- ggplot(
  test_bins,
  aes(
    x = mean_predicted,
    y = observed_fraction,
    linetype = calibration,
    shape = calibration
  )
) +
  geom_abline(linetype = 2) +
  geom_line() +
  geom_point(size = 2) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(
    title = "Ensemble calibration on held-out test data",
    x = "Mean predicted probability",
    y = "Observed symptomatic-influenza proportion",
    linetype = NULL,
    shape = NULL
  ) +
  theme_minimal()

ggsave(
  "results/ensemble_calibration_test_before_after.png",
  test_plot,
  width = 7,
  height = 6,
  dpi = 300
)

# ------------------------------------------------------------
# 10. Print results
# ------------------------------------------------------------

cat("\nCalibration analysis completed.\n\n")

cat("Training calibration before scaling:\n")
print(training_metrics)

cat("\nEnsemble training calibration before and after scaling:\n")
print(ensemble_training_comparison)

cat("\nHeld-out test ensemble calibration before and after scaling:\n")
print(test_metrics)

cat(
  "\nNote: the Hosmer-Lemeshow test is sensitive to sample size ",
  "and grouping, so interpret it together with calibration plots, ",
  "Brier score, intercept and slope.\n",
  sep = ""
)
