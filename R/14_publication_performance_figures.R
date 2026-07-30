# R/14_publication_performance_figures.R
# Publication-style ROC and calibration figures.
#
# Outputs:
#   Figure_2_ROC_test.png / .tiff
#   Figure_2_ROC_training_OOF.png / .tiff
#   Figure_3_calibration_test.png / .tiff
#   Figure_3_calibration_training_OOF.png / .tiff
#
# ROC confidence bands use stratified bootstrap resampling.
# Calibration confidence bands use stratified bootstrap logistic-spline fits.

library(dplyr)
library(tidyr)
library(ggplot2)
library(pROC)
library(splines)

dir.create("results", showWarnings = FALSE)

# Increase to 2000 for the final publication export if desired.
roc_boot_n <- 1000
calibration_boot_n <- 1000
random_seed <- 2026

# ------------------------------------------------------------
# 1. General helpers
# ------------------------------------------------------------

clip_probability <- function(p, epsilon = 1e-6) {
  pmin(pmax(as.numeric(p), epsilon), 1 - epsilon)
}

brier_score <- function(y, p) {
  mean((as.numeric(y) - as.numeric(p))^2)
}

calibration_intercept_slope <- function(y, p) {
  p <- clip_probability(p)
  linear_predictor <- qlogis(p)

  intercept_fit <- glm(
    y ~ 1,
    offset = linear_predictor,
    family = binomial()
  )

  slope_fit <- glm(
    y ~ linear_predictor,
    family = binomial()
  )

  c(
    intercept = unname(coef(intercept_fit)[1]),
    slope = unname(coef(slope_fit)["linear_predictor"])
  )
}

publication_theme <- theme_classic(base_size = 11) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(
      face = "bold",
      size = 11,
      hjust = 0
    ),
    axis.title = element_text(face = "bold"),
    panel.spacing = grid::unit(1.2, "lines"),
    plot.title = element_text(face = "bold"),
    legend.position = "none"
  )

save_publication_figure <- function(plot, filename, width, height) {
  ggsave(
    paste0("results/", filename, ".png"),
    plot,
    width = width,
    height = height,
    dpi = 600,
    bg = "white"
  )

  ggsave(
    paste0("results/", filename, ".tiff"),
    plot,
    width = width,
    height = height,
    dpi = 600,
    compression = "lzw",
    bg = "white"
  )
}

# ------------------------------------------------------------
# 2. ROC figure function
# ------------------------------------------------------------

create_roc_figure <- function(
  data,
  observed_column,
  probability_columns,
  model_labels,
  figure_title,
  output_name,
  boot_n = 1000
) {

  observed_name <- rlang::ensym(observed_column)

  observed_raw <- data %>% pull(!!observed_name)

  if (is.numeric(observed_raw)) {
    observed <- factor(
      ifelse(observed_raw == 1, "Positive", "Negative"),
      levels = c("Negative", "Positive")
    )
  } else {
    observed <- factor(
      observed_raw,
      levels = c("Negative", "Positive")
    )
  }

  curve_list <- list()
  band_list <- list()
  auc_list <- list()

  set.seed(random_seed)

  for (i in seq_along(probability_columns)) {

    probability <- as.numeric(data[[probability_columns[i]]])
    model_name <- model_labels[i]

    roc_object <- roc(
      response = observed,
      predictor = probability,
      levels = c("Negative", "Positive"),
      direction = "<",
      quiet = TRUE
    )

    curve_coordinates <- coords(
      roc_object,
      x = "all",
      ret = c("specificity", "sensitivity"),
      transpose = FALSE
    ) %>%
      as.data.frame() %>%
      transmute(
        specificity = specificity * 100,
        sensitivity = sensitivity * 100,
        model = model_name
      )

    specificity_grid <- seq(0, 1, by = 0.01)

    sensitivity_ci <- ci.se(
      roc_object,
      specificities = specificity_grid,
      boot.n = boot_n,
      boot.stratified = TRUE,
      progress = "none"
    )

    ci_frame <- as.data.frame(sensitivity_ci)
    ci_columns <- names(ci_frame)

    band_data <- data.frame(
      specificity = specificity_grid * 100,
      lower = ci_frame[[ci_columns[1]]] * 100,
      median = ci_frame[[ci_columns[2]]] * 100,
      upper = ci_frame[[ci_columns[3]]] * 100,
      model = model_name
    )

    auc_ci <- ci.auc(
      roc_object,
      method = "bootstrap",
      boot.n = boot_n,
      boot.stratified = TRUE
    )

    auc_value <- as.numeric(auc(roc_object))

    auc_information <- data.frame(
      model = model_name,
      x = 96,
      y = 7,
      label = sprintf(
        "AUC %.3f\n95%% CI %.3f–%.3f",
        auc_value,
        as.numeric(auc_ci[1]),
        as.numeric(auc_ci[3])
      )
    )

    curve_list[[i]] <- curve_coordinates
    band_list[[i]] <- band_data
    auc_list[[i]] <- auc_information
  }

  curve_data <- bind_rows(curve_list)
  band_data <- bind_rows(band_list)
  auc_data <- bind_rows(auc_list)

  model_factor_levels <- model_labels

  curve_data$model <- factor(
    curve_data$model,
    levels = model_factor_levels
  )

  band_data$model <- factor(
    band_data$model,
    levels = model_factor_levels
  )

  auc_data$model <- factor(
    auc_data$model,
    levels = model_factor_levels
  )

  roc_plot <- ggplot() +
    geom_ribbon(
      data = band_data,
      aes(
        x = specificity,
        ymin = lower,
        ymax = upper
      ),
      fill = "grey75",
      alpha = 0.55
    ) +
    geom_line(
      data = curve_data,
      aes(
        x = specificity,
        y = sensitivity
      ),
      linewidth = 0.8
    ) +
    geom_abline(
      intercept = 100,
      slope = -1,
      linetype = 3,
      linewidth = 0.6
    ) +
    geom_text(
      data = auc_data,
      aes(
        x = x,
        y = y,
        label = label
      ),
      hjust = 0,
      vjust = 0,
      size = 3
    ) +
    facet_wrap(
      ~ model,
      ncol = 2
    ) +
    scale_x_reverse(
      limits = c(100, 0),
      breaks = seq(100, 0, by = -20)
    ) +
    scale_y_continuous(
      limits = c(0, 100),
      breaks = seq(0, 100, by = 20)
    ) +
    coord_equal() +
    labs(
      title = figure_title,
      subtitle = paste0(
        "Shaded regions show stratified bootstrap 95% confidence intervals (",
        format(boot_n, big.mark = ","),
        " replicates)"
      ),
      x = "Specificity (%)",
      y = "Sensitivity (%)"
    ) +
    publication_theme

  save_publication_figure(
    roc_plot,
    output_name,
    width = 8.2,
    height = 7.3
  )

  roc_plot
}

# ------------------------------------------------------------
# 3. Calibration curve with bootstrap confidence interval
# ------------------------------------------------------------

fit_spline_calibration <- function(y, p, prediction_grid) {
  p <- clip_probability(p)

  model <- glm(
    y ~ ns(p, df = 3),
    family = binomial(),
    control = glm.control(maxit = 100)
  )

  prediction_frame <- data.frame(
    p = prediction_grid
  )

  as.numeric(
    predict(
      model,
      newdata = prediction_frame,
      type = "response"
    )
  )
}

bootstrap_calibration_curve <- function(
  y,
  p,
  prediction_grid = seq(0.01, 0.99, length.out = 100),
  boot_n = 1000,
  seed = 2026
) {

  y <- as.numeric(y)
  p <- clip_probability(p)

  fitted_curve <- fit_spline_calibration(
    y,
    p,
    prediction_grid
  )

  positive_indices <- which(y == 1)
  negative_indices <- which(y == 0)

  bootstrap_predictions <- matrix(
    NA_real_,
    nrow = boot_n,
    ncol = length(prediction_grid)
  )

  set.seed(seed)

  successful_fits <- 0

  for (b in seq_len(boot_n)) {

    sampled_indices <- c(
      sample(
        positive_indices,
        length(positive_indices),
        replace = TRUE
      ),
      sample(
        negative_indices,
        length(negative_indices),
        replace = TRUE
      )
    )

    bootstrap_fit <- tryCatch(
      fit_spline_calibration(
        y[sampled_indices],
        p[sampled_indices],
        prediction_grid
      ),
      error = function(e) NULL,
      warning = function(w) {
        suppressWarnings(
          tryCatch(
            fit_spline_calibration(
              y[sampled_indices],
              p[sampled_indices],
              prediction_grid
            ),
            error = function(e) NULL
          )
        )
      }
    )

    if (!is.null(bootstrap_fit) &&
        all(is.finite(bootstrap_fit))) {
      successful_fits <- successful_fits + 1
      bootstrap_predictions[successful_fits, ] <- bootstrap_fit
    }
  }

  bootstrap_predictions <- bootstrap_predictions[
    seq_len(successful_fits),
    ,
    drop = FALSE
  ]

  lower <- apply(
    bootstrap_predictions,
    2,
    quantile,
    probs = 0.025,
    na.rm = TRUE
  )

  upper <- apply(
    bootstrap_predictions,
    2,
    quantile,
    probs = 0.975,
    na.rm = TRUE
  )

  data.frame(
    predicted = prediction_grid,
    observed = fitted_curve,
    lower = lower,
    upper = upper
  )
}

create_calibration_figure <- function(
  data,
  observed_column,
  before_column,
  after_column,
  figure_title,
  output_name,
  boot_n = 1000
) {

  observed_name <- rlang::ensym(observed_column)
  observed_raw <- data %>% pull(!!observed_name)

  if (is.numeric(observed_raw)) {
    observed <- as.numeric(observed_raw)
  } else {
    observed <- ifelse(observed_raw == "Positive", 1, 0)
  }

  calibration_stages <- list(
    "a  Unscaled calibration" =
      as.numeric(data[[before_column]]),
    "b  Isotonic-scaled calibration" =
      as.numeric(data[[after_column]])
  )

  curve_list <- list()
  metric_list <- list()

  for (i in seq_along(calibration_stages)) {

    stage_name <- names(calibration_stages)[i]
    probability <- calibration_stages[[i]]

    curve <- bootstrap_calibration_curve(
      y = observed,
      p = probability,
      boot_n = boot_n,
      seed = random_seed + i
    ) %>%
      mutate(stage = stage_name)

    calibration_values <- calibration_intercept_slope(
      observed,
      probability
    )

    metrics <- data.frame(
      stage = stage_name,
      x = 0.04,
      y = 0.96,
      label = sprintf(
        "Intercept %.3f\nSlope %.3f\nBrier %.3f",
        calibration_values["intercept"],
        calibration_values["slope"],
        brier_score(observed, probability)
      )
    )

    curve_list[[i]] <- curve
    metric_list[[i]] <- metrics
  }

  curve_data <- bind_rows(curve_list)
  metric_data <- bind_rows(metric_list)

  stage_levels <- names(calibration_stages)

  curve_data$stage <- factor(
    curve_data$stage,
    levels = stage_levels
  )

  metric_data$stage <- factor(
    metric_data$stage,
    levels = stage_levels
  )

  calibration_plot <- ggplot(
    curve_data,
    aes(
      x = predicted,
      y = observed
    )
  ) +
    geom_ribbon(
      aes(
        ymin = lower,
        ymax = upper
      ),
      fill = "grey75",
      alpha = 0.6
    ) +
    geom_line(linewidth = 0.9) +
    geom_abline(
      intercept = 0,
      slope = 1,
      linetype = 3,
      linewidth = 0.6
    ) +
    geom_text(
      data = metric_data,
      aes(
        x = x,
        y = y,
        label = label
      ),
      inherit.aes = FALSE,
      hjust = 0,
      vjust = 1,
      size = 3
    ) +
    facet_wrap(
      ~ stage,
      nrow = 1
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.2)
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, by = 0.2)
    ) +
    coord_equal() +
    labs(
      title = figure_title,
      subtitle = paste0(
        "Grey regions show stratified bootstrap 95% confidence intervals (",
        format(boot_n, big.mark = ","),
        " replicates)"
      ),
      x = "Predicted probability",
      y = "Observed symptomatic-influenza proportion"
    ) +
    publication_theme

  save_publication_figure(
    calibration_plot,
    output_name,
    width = 8.4,
    height = 4.8
  )

  calibration_plot
}

# ------------------------------------------------------------
# 4. Figure 2: held-out test ROC curves
# ------------------------------------------------------------

test_predictions <- read.csv(
  "results/external_test_predictions.csv",
  stringsAsFactors = FALSE
)

test_roc_plot <- create_roc_figure(
  data = test_predictions,
  observed_column = observed,
  probability_columns = c(
    "elastic_probability",
    "rf_probability",
    "xgb_probability",
    "ensemble_probability"
  ),
  model_labels = c(
    "a  Elastic Net",
    "b  Random Forest",
    "c  XGBoost",
    "d  Equal-weight ensemble"
  ),
  figure_title = "Model discrimination in the held-out test set",
  output_name = "Figure_2_ROC_test",
  boot_n = roc_boot_n
)

# ------------------------------------------------------------
# 5. Supplementary ROC figure: out-of-fold training predictions
# ------------------------------------------------------------

training_predictions <- read.csv(
  "results/ensemble_cv_predictions.csv",
  stringsAsFactors = FALSE
)

training_roc_plot <- create_roc_figure(
  data = training_predictions,
  observed_column = observed,
  probability_columns = c(
    "elastic_probability",
    "rf_probability",
    "xgb_probability",
    "ensemble_probability"
  ),
  model_labels = c(
    "a  Elastic Net",
    "b  Random Forest",
    "c  XGBoost",
    "d  Equal-weight ensemble"
  ),
  figure_title = "Model discrimination using out-of-fold training predictions",
  output_name = "Figure_2_ROC_training_OOF",
  boot_n = roc_boot_n
)

# ------------------------------------------------------------
# 6. Figure 3: held-out test calibration before/after isotonic scaling
# ------------------------------------------------------------

test_calibration_data <- read.csv(
  "results/external_test_predictions_calibrated.csv",
  stringsAsFactors = FALSE
)

test_calibration_plot <- create_calibration_figure(
  data = test_calibration_data,
  observed_column = observed,
  before_column = "ensemble_probability",
  after_column = "ensemble_probability_calibrated",
  figure_title = "Ensemble calibration in the held-out test set",
  output_name = "Figure_3_calibration_test",
  boot_n = calibration_boot_n
)

# ------------------------------------------------------------
# 7. Supplementary calibration figure: cross-fitted training scaling
# ------------------------------------------------------------

training_calibration_data <- read.csv(
  "results/calibration_training_predictions.csv",
  stringsAsFactors = FALSE
)

training_calibration_plot <- create_calibration_figure(
  data = training_calibration_data,
  observed_column = observed,
  before_column = "Equal_weight_ensemble",
  after_column = "ensemble_isotonic_crossfit",
  figure_title = "Ensemble calibration using out-of-fold training predictions",
  output_name = "Figure_3_calibration_training_OOF",
  boot_n = calibration_boot_n
)

cat("\nPublication figures completed.\n\n")

cat("Main ROC figure:\n")
cat("  results/Figure_2_ROC_test.png\n")
cat("  results/Figure_2_ROC_test.tiff\n\n")

cat("Main calibration figure:\n")
cat("  results/Figure_3_calibration_test.png\n")
cat("  results/Figure_3_calibration_test.tiff\n\n")

cat("Additional training figures:\n")
cat("  results/Figure_2_ROC_training_OOF.png\n")
cat("  results/Figure_3_calibration_training_OOF.png\n\n")

cat(
  "These figures use ordinary stratified bootstrap confidence bands, ",
  "not the 0.632 bootstrap procedure used in the example paper.\n",
  sep = ""
)
