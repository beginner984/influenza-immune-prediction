# R/13_variable_importance.R
# Calculate comparable percentage variable importance for:
#   1. Elastic Net
#   2. Random Forest
#   3. XGBoost
#   4. Equal-weight ensemble
#
# Ensemble importance is the equal-weight average of the three
# normalized component-model importance percentages.

library(caret)
library(dplyr)
library(tidyr)
library(ggplot2)
library(glmnet)

dir.create("results", showWarnings = FALSE)

# ------------------------------------------------------------
# 1. Helper function
# ------------------------------------------------------------

normalize_to_percentage <- function(values) {
  values <- as.numeric(values)
  values[is.na(values)] <- 0

  # Negative permutation importance indicates no reliable contribution.
  # Set negative values to zero before expressing scores as percentages.
  values[values < 0] <- 0

  if (sum(values) == 0) {
    return(rep(0, length(values)))
  }

  100 * values / sum(values)
}

# ------------------------------------------------------------
# 2. Elastic Net importance and odds ratios
# ------------------------------------------------------------

elastic_model <- readRDS("models/elastic_net_model.rds")

elastic_coefficients <- as.matrix(
  coef(
    elastic_model$finalModel,
    s = elastic_model$bestTune$lambda
  )
)

elastic_table <- data.frame(
  variable = rownames(elastic_coefficients),
  coefficient = as.numeric(elastic_coefficients[, 1]),
  stringsAsFactors = FALSE
) %>%
  filter(variable != "(Intercept)") %>%
  mutate(
    absolute_coefficient = abs(coefficient),
    odds_ratio_per_1_SD = exp(coefficient)
  )

# These coefficients correspond to standardized predictors because
# center/scale preprocessing was used during Elastic Net training.
write.csv(
  elastic_table,
  "results/elastic_net_coefficients_odds_ratios.csv",
  row.names = FALSE
)

elastic_importance <- elastic_table %>%
  transmute(
    variable,
    Elastic_Net_raw = absolute_coefficient,
    Elastic_Net = normalize_to_percentage(
      absolute_coefficient
    )
  )

# ------------------------------------------------------------
# 3. Random Forest permutation importance
# ------------------------------------------------------------

rf_model <- readRDS("models/random_forest_model.rds")

rf_raw <- varImp(
  rf_model,
  scale = FALSE
)$importance %>%
  tibble::rownames_to_column("variable")

rf_importance <- rf_raw %>%
  transmute(
    variable,
    Random_Forest_raw = Overall,
    Random_Forest = normalize_to_percentage(Overall)
  )

# ------------------------------------------------------------
# 4. XGBoost gain importance
# ------------------------------------------------------------

xgb_raw <- read.csv(
  "results/xgboost_native_variable_importance.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

xgb_importance <- xgb_raw %>%
  transmute(
    variable = Feature,
    XGBoost_raw = Gain,
    XGBoost = normalize_to_percentage(Gain)
  )

# ------------------------------------------------------------
# 5. Combine all 16 predictors
# ------------------------------------------------------------

all_predictors <- data.frame(
  variable = elastic_importance$variable,
  stringsAsFactors = FALSE
)

importance_table <- all_predictors %>%
  left_join(elastic_importance, by = "variable") %>%
  left_join(rf_importance, by = "variable") %>%
  left_join(xgb_importance, by = "variable") %>%
  mutate(
    across(
      c(
        Elastic_Net_raw,
        Elastic_Net,
        Random_Forest_raw,
        Random_Forest,
        XGBoost_raw,
        XGBoost
      ),
      ~ replace_na(.x, 0)
    ),
    Equal_weight_ensemble = (
      Elastic_Net +
      Random_Forest +
      XGBoost
    ) / 3
  ) %>%
  arrange(desc(Equal_weight_ensemble))

write.csv(
  importance_table,
  "results/variable_importance_percentages.csv",
  row.names = FALSE
)

# A publication-friendly table with percentages only.
importance_percentages <- importance_table %>%
  select(
    variable,
    Elastic_Net,
    Random_Forest,
    XGBoost,
    Equal_weight_ensemble
  ) %>%
  mutate(
    across(
      -variable,
      ~ round(.x, 2)
    )
  )

write.csv(
  importance_percentages,
  "results/variable_importance_table_publication.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 6. Long-format data for plots
# ------------------------------------------------------------

importance_long <- importance_percentages %>%
  pivot_longer(
    cols = -variable,
    names_to = "model",
    values_to = "importance"
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

variable_order <- importance_percentages$variable

importance_long$variable <- factor(
  importance_long$variable,
  levels = rev(variable_order)
)

# ------------------------------------------------------------
# 7. Heatmap of all predictors and models
# ------------------------------------------------------------

heatmap_plot <- ggplot(
  importance_long,
  aes(x = model, y = variable, fill = importance)
) +
  geom_tile() +
  geom_text(
    aes(label = sprintf("%.1f", importance)),
    size = 3
  ) +
  labs(
    title = "Variable importance across models",
    subtitle = "Values are percentage contributions within each model",
    x = NULL,
    y = NULL,
    fill = "Importance (%)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 25,
      hjust = 1
    )
  )

ggsave(
  "results/variable_importance_heatmap.png",
  heatmap_plot,
  width = 10,
  height = 8,
  dpi = 300
)

# ------------------------------------------------------------
# 8. Grouped bar chart for top 10 ensemble predictors
# ------------------------------------------------------------

top_10_variables <- importance_percentages %>%
  slice_head(n = 10) %>%
  pull(variable)

top_10_long <- importance_long %>%
  filter(as.character(variable) %in% top_10_variables)

top_10_long$variable <- factor(
  as.character(top_10_long$variable),
  levels = rev(top_10_variables)
)

grouped_plot <- ggplot(
  top_10_long,
  aes(
    x = importance,
    y = variable,
    fill = model
  )
) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.7
  ) +
  labs(
    title = "Top 10 predictors across models",
    x = "Variable importance (%)",
    y = NULL,
    fill = NULL
  ) +
  theme_minimal()

ggsave(
  "results/variable_importance_top10_grouped.png",
  grouped_plot,
  width = 10,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# 9. Ensemble-only variable importance plot
# ------------------------------------------------------------

ensemble_plot_data <- importance_percentages %>%
  mutate(
    variable = factor(
      variable,
      levels = rev(variable)
    )
  )

ensemble_plot <- ggplot(
  ensemble_plot_data,
  aes(
    x = Equal_weight_ensemble,
    y = variable
  )
) +
  geom_col() +
  labs(
    title = "Equal-weight ensemble variable importance",
    subtitle = "Average normalized contribution of Elastic Net, RF and XGBoost",
    x = "Variable importance (%)",
    y = NULL
  ) +
  theme_minimal()

ggsave(
  "results/ensemble_variable_importance.png",
  ensemble_plot,
  width = 8,
  height = 7,
  dpi = 300
)

# ------------------------------------------------------------
# 10. Print top variables
# ------------------------------------------------------------

cat("\nVariable-importance analysis completed.\n\n")

cat("Top 10 predictors by ensemble percentage contribution:\n")
print(
  importance_percentages %>%
    select(
      variable,
      Elastic_Net,
      Random_Forest,
      XGBoost,
      Equal_weight_ensemble
    ) %>%
    slice_head(n = 10)
)

cat(
  "\nElastic Net coefficients and odds ratios were also saved. ",
  "The odds ratios represent a one-standard-deviation increase ",
  "in each predictor.\n",
  sep = ""
)
