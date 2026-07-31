# R/17_visualise_vaccination_comparison.R
#
# Visual comparison of:
#   1. Original models: 16 immune predictors
#   2. Sensitivity models: 16 immune predictors + vaccination
#
# This script reads the real cross-validation results already produced.
# It does not contain manually entered performance values.

library(dplyr)
library(tidyr)
library(ggplot2)

# ------------------------------------------------------------
# 1. Read the real model-comparison results
# ------------------------------------------------------------

input_file <- paste0(
  "results/",
  "primary_vs_vaccination_sensitivity_cv.csv"
)

if (!file.exists(input_file)) {
  stop(
    "The results file was not found: ",
    input_file
  )
}

comparison_data <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

required_columns <- c(
  "analysis",
  "model",
  "ROC_AUC",
  "sensitivity",
  "specificity",
  "balanced_accuracy",
  "accuracy"
)

missing_columns <- setdiff(
  required_columns,
  names(comparison_data)
)

if (length(missing_columns) > 0) {
  stop(
    "Missing columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

# ------------------------------------------------------------
# 2. Create clear analysis labels
# ------------------------------------------------------------

comparison_data <- comparison_data %>%
  mutate(
    analysis_key = case_when(
      grepl(
        "^Primary",
        analysis,
        ignore.case = TRUE
      ) ~ "Primary",
      
      grepl(
        "^Post-hoc",
        analysis,
        ignore.case = TRUE
      ) ~ "Vaccination",
      
      TRUE ~ NA_character_
    )
  )

if (any(is.na(comparison_data$analysis_key))) {
  stop(
    "Some analysis labels could not be recognised."
  )
}

model_order <- c(
  "Elastic Net",
  "Random Forest",
  "XGBoost",
  "Equal-weight ensemble"
)

comparison_data$model <- factor(
  comparison_data$model,
  levels = model_order
)

comparison_data$analysis_key <- factor(
  comparison_data$analysis_key,
  levels = c(
    "Primary",
    "Vaccination"
  ),
  labels = c(
    "Original: 16 immune predictors",
    "16 immune predictors + vaccination"
  )
)

# ------------------------------------------------------------
# 3. Convert performance measures to long format
# ------------------------------------------------------------

performance_long <- comparison_data %>%
  select(
    analysis_key,
    model,
    ROC_AUC,
    sensitivity,
    specificity,
    balanced_accuracy,
    accuracy
  ) %>%
  pivot_longer(
    cols = c(
      ROC_AUC,
      sensitivity,
      specificity,
      balanced_accuracy,
      accuracy
    ),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = recode(
      metric,
      ROC_AUC = "ROC AUC",
      sensitivity = "Sensitivity",
      specificity = "Specificity",
      balanced_accuracy = "Balanced accuracy",
      accuracy = "Accuracy"
    )
  )

# ------------------------------------------------------------
# 4. Figure 1:
#    Actual performance before and after vaccination
# ------------------------------------------------------------

performance_plot <- ggplot(
  performance_long,
  aes(
    x = value,
    y = model
  )
) +
  geom_line(
    aes(group = model),
    linewidth = 0.7,
    colour = "grey65"
  ) +
  geom_point(
    aes(shape = analysis_key),
    size = 3
  ) +
  facet_wrap(
    ~ metric,
    ncol = 2
  ) +
  scale_shape_manual(
    values = c(16, 17)
  ) +
  scale_x_continuous(
    limits = c(0.60, 1.00),
    breaks = seq(
      0.60,
      1.00,
      by = 0.10
    )
  ) +
  labs(
    title = paste0(
      "Model performance before and after ",
      "adding vaccination status"
    ),
    subtitle = paste0(
      "Repeated cross-validation using the ",
      "134-participant training cohort"
    ),
    x = "Cross-validated performance",
    y = NULL,
    shape = NULL,
    caption = paste0(
      "Lines connect the original and ",
      "vaccination-adjusted version of each model."
    )
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    strip.text = element_text(
      face = "bold"
    )
  )

print(performance_plot)

ggsave(
  filename = paste0(
    "results/",
    "Figure_4_primary_vs_vaccination_performance.png"
  ),
  plot = performance_plot,
  width = 10,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# 5. Calculate the exact change caused by vaccination
# ------------------------------------------------------------

change_data <- performance_long %>%
  mutate(
    analysis_short = case_when(
      analysis_key ==
        "Original: 16 immune predictors" ~
        "Primary",
      
      analysis_key ==
        "16 immune predictors + vaccination" ~
        "Vaccination",
      
      TRUE ~ NA_character_
    )
  ) %>%
  select(
    model,
    metric,
    analysis_short,
    value
  ) %>%
  pivot_wider(
    names_from = analysis_short,
    values_from = value
  ) %>%
  mutate(
    change = Vaccination - Primary,
    change_percentage_points = change * 100
  )

# ------------------------------------------------------------
# 6. Figure 2:
#    Exact change after adding vaccination
# ------------------------------------------------------------

change_plot <- ggplot(
  change_data,
  aes(
    x = change_percentage_points,
    y = model
  )
) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.7
  ) +
  geom_col(
    width = 0.6
  ) +
  geom_text(
    aes(
      label = sprintf(
        "%+.2f pp",
        change_percentage_points
      ),
      hjust = ifelse(
        change_percentage_points >= 0,
        -0.15,
        1.15
      )
    ),
    size = 3.5
  ) +
  facet_wrap(
    ~ metric,
    ncol = 2,
    scales = "free_x"
  ) +
  scale_x_continuous(
    expand = expansion(
      mult = c(0.30, 0.30)
    )
  ) +
  labs(
    title = paste0(
      "Change in performance after adding ",
      "vaccination status"
    ),
    subtitle = paste0(
      "Positive values indicate improvement; ",
      "negative values indicate reduction"
    ),
    x = "Change in percentage points",
    y = NULL,
    caption = paste0(
      "Calculated directly from ",
      "primary_vs_vaccination_sensitivity_cv.csv"
    )
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(
      face = "bold"
    )
  )

print(change_plot)

ggsave(
  filename = paste0(
    "results/",
    "Figure_5_vaccination_performance_change.png"
  ),
  plot = change_plot,
  width = 10,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# 7. Save and print the exact numerical changes
# ------------------------------------------------------------

write.csv(
  change_data,
  paste0(
    "results/",
    "vaccination_performance_change_table.csv"
  ),
  row.names = FALSE
)

cat(
  "\nVaccination comparison figures created.\n\n"
)

cat(
  "Figure 1:\n",
  "results/Figure_4_primary_vs_vaccination_performance.png\n\n"
)

cat(
  "Figure 2:\n",
  "results/Figure_5_vaccination_performance_change.png\n\n"
)

cat(
  "Numerical changes:\n"
)

print(
  change_data %>%
    arrange(
      metric,
      model
    )
)