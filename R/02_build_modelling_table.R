# R/02_build_modelling_table.R
# Create one validated modelling table with 16 predictors.

library(dplyr)

# 1. Input and output files
input_file <- "data/processed/Mettelman_cleaned.csv"
output_file <- "data/processed/modelling_data_16.csv"
missing_file <- "data/processed/modelling_missing_report.csv"
outcome_file <- "data/processed/modelling_outcome_balance.csv"

# 2. Read the cleaned dataset
data <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = TRUE
)

# 3. Define the 16 predictors
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

# 4. Check that every required column exists
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

# 5. Keep only ID, split label, outcome and 16 predictors
model_data <- data %>%
  select(all_of(required_columns))

# 6. Convert variables to appropriate types
model_data <- model_data %>%
  mutate(
    Study.ID = as.character(Study.ID),
    training = factor(training, levels = c("training", "testing")),
    PCRpositive = factor(
      PCRpositive,
      levels = c("Negative", "Positive")
    ),
    across(
      all_of(selected_predictors),
      ~ suppressWarnings(as.numeric(as.character(.x)))
    )
  )

# 7. Check participant IDs
duplicate_ids <- model_data %>%
  count(Study.ID, name = "n") %>%
  filter(n > 1)

if (nrow(duplicate_ids) > 0) {
  warning("Duplicate participant IDs were found.")
}

# 8. Check train/test labels and outcome levels
if (any(is.na(model_data$training))) {
  warning("Some rows do not have a valid training/testing label.")
}

if (nlevels(droplevels(model_data$PCRpositive)) != 2) {
  warning("The outcome does not contain both Positive and Negative classes.")
}

# 9. Create a missing-value report
missing_report <- data.frame(
  variable = names(model_data),
  missing_n = sapply(model_data, function(x) sum(is.na(x))),
  missing_percent = round(
    100 * sapply(model_data, function(x) mean(is.na(x))),
    2
  ),
  row.names = NULL
)

# 10. Check class balance separately in training and testing data
outcome_balance <- model_data %>%
  count(training, PCRpositive, name = "participants") %>%
  group_by(training) %>%
  mutate(percent = round(100 * participants / sum(participants), 1)) %>%
  ungroup()

# 11. Save the modelling table and audit reports
write.csv(model_data, output_file, row.names = FALSE)
write.csv(missing_report, missing_file, row.names = FALSE)
write.csv(outcome_balance, outcome_file, row.names = FALSE)

# 12. Print a short summary
cat("\nModelling table created successfully.\n")
cat("Participants:", nrow(model_data), "\n")
cat("Columns:", ncol(model_data), "\n")
cat("Predictors:", length(selected_predictors), "\n\n")

cat("Outcome balance:\n")
print(outcome_balance)

cat("\nVariables with missing values:\n")
print(missing_report %>% filter(missing_n > 0))

cat("\nSaved:", output_file, "\n")
