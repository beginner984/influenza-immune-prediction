# R/04_prepare_train_test.R
# Separate the complete-case data into training and test sets,
# then audit the 16 predictors using TRAINING DATA ONLY.
# This script does not train any model.

library(dplyr)
library(caret)

# ------------------------------------------------------------
# 1. File locations
# ------------------------------------------------------------

input_file <- "data/processed/modelling_data_16_complete.csv"
training_file <- "data/processed/training_data_16.csv"
testing_file <- "data/processed/testing_data_16.csv"

nzv_file <- "data/processed/near_zero_variance_report.csv"
correlation_file <- "data/processed/high_correlation_pairs.csv"
summary_file <- "data/processed/training_predictor_summary.csv"


# ------------------------------------------------------------
# 2. Read the complete-case dataset
# ------------------------------------------------------------

data <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = TRUE
)


# ------------------------------------------------------------
# 3. Define the same 16 predictors
# ------------------------------------------------------------

selected_predictors <- c(
  "cTfh.ICOSp",
  "CD4.Effector",
  "cTfh.CXCR3p",
  "CD4.IL2",
  "CD8.Effector",
  "NK.Activated",
  "CD8.Memory.CCR5p",
  "NK.GZBp.IFNp",
  "CD4.Naive",
  "nai_h3",
  "CD8.SP",
  "CD4.DP",
  "NK.CK.Producing",
  "CD8.DP",
  "Gamma.Delta",
  "CD8.Tcells"
)


# ------------------------------------------------------------
# 4. Validate and convert data types
# ------------------------------------------------------------

required_columns <- c(
  "Study.ID",
  "training",
  "PCRpositive",
  selected_predictors
)

missing_columns <- setdiff(required_columns, names(data))

if (length(missing_columns) > 0) {
  stop(
    "Required columns are missing: ",
    paste(missing_columns, collapse = ", ")
  )
}

data <- data %>%
  mutate(
    Study.ID = as.character(Study.ID),
    training = as.character(training),

    # Positive is placed first because it is the event of interest.
    PCRpositive = factor(
      PCRpositive,
      levels = c("Positive", "Negative")
    ),

    # All 16 predictors must be numeric.
    across(
      all_of(selected_predictors),
      ~ suppressWarnings(as.numeric(as.character(.x)))
    )
  )

# Stop if type conversion created any missing values.
if (anyNA(data[, selected_predictors])) {
  stop("At least one predictor became NA after numeric conversion.")
}


# ------------------------------------------------------------
# 5. Separate training and test participants
# ------------------------------------------------------------

training_data <- data %>%
  filter(training == "training") %>%
  select(Study.ID, PCRpositive, all_of(selected_predictors))

testing_data <- data %>%
  filter(training == "testing") %>%
  select(Study.ID, PCRpositive, all_of(selected_predictors))

# Confirm that no participant appears in both sets.
overlapping_ids <- intersect(
  training_data$Study.ID,
  testing_data$Study.ID
)

if (length(overlapping_ids) > 0) {
  stop("Some participant IDs appear in both training and test sets.")
}


# ------------------------------------------------------------
# 6. Check class balance
# ------------------------------------------------------------

training_balance <- prop.table(table(training_data$PCRpositive))
testing_balance <- prop.table(table(testing_data$PCRpositive))


# ------------------------------------------------------------
# 7. Check near-zero-variance predictors
#    IMPORTANT: use training data only
# ------------------------------------------------------------

nzv_metrics <- nearZeroVar(
  training_data[, selected_predictors],
  saveMetrics = TRUE
)

nzv_report <- data.frame(
  variable = rownames(nzv_metrics),
  nzv_metrics,
  row.names = NULL
)


# ------------------------------------------------------------
# 8. Check strong correlations
#    IMPORTANT: use training data only
# ------------------------------------------------------------

correlation_matrix <- cor(
  training_data[, selected_predictors],
  use = "pairwise.complete.obs",
  method = "pearson"
)

# Keep each variable pair only once.
pair_index <- which(
  abs(correlation_matrix) > 0.90 &
    upper.tri(correlation_matrix),
  arr.ind = TRUE
)

if (nrow(pair_index) == 0) {
  high_correlation_pairs <- data.frame(
    variable_1 = character(0),
    variable_2 = character(0),
    correlation = numeric(0)
  )
} else {
  high_correlation_pairs <- data.frame(
    variable_1 = rownames(correlation_matrix)[pair_index[, 1]],
    variable_2 = colnames(correlation_matrix)[pair_index[, 2]],
    correlation = correlation_matrix[pair_index]
  ) %>%
    arrange(desc(abs(correlation)))
}


# ------------------------------------------------------------
# 9. Summarise training predictor distributions
# ------------------------------------------------------------

training_summary <- data.frame(
  variable = selected_predictors,
  minimum = sapply(training_data[, selected_predictors], min),
  median = sapply(training_data[, selected_predictors], median),
  mean = sapply(training_data[, selected_predictors], mean),
  maximum = sapply(training_data[, selected_predictors], max),
  standard_deviation = sapply(training_data[, selected_predictors], sd),
  row.names = NULL
)


# ------------------------------------------------------------
# 10. Save the prepared sets and audit reports
# ------------------------------------------------------------

write.csv(training_data, training_file, row.names = FALSE)
write.csv(testing_data, testing_file, row.names = FALSE)
write.csv(nzv_report, nzv_file, row.names = FALSE)
write.csv(high_correlation_pairs, correlation_file, row.names = FALSE)
write.csv(training_summary, summary_file, row.names = FALSE)


# ------------------------------------------------------------
# 11. Print a concise report
# ------------------------------------------------------------

cat("\nTrain/test preparation completed.\n")
cat("Training participants:", nrow(training_data), "\n")
cat("Testing participants:", nrow(testing_data), "\n\n")

cat("Training outcome proportions:\n")
print(round(100 * training_balance, 1))

cat("\nTesting outcome proportions:\n")
print(round(100 * testing_balance, 1))

cat("\nNear-zero-variance predictors:\n")
print(nzv_report %>% filter(nzv == TRUE))

cat("\nPredictor pairs with |correlation| > 0.90:\n")
print(high_correlation_pairs)

cat("\nNo model has been trained yet.\n")
