# R/03_create_complete_case_data.R
# Create a modelling dataset containing only participants
# with complete values for all 16 predictors.

library(dplyr)
library(tidyr)

# 1. Read the 200-participant modelling table
input_file <- "data/processed/modelling_data_16.csv"
output_file <- "data/processed/modelling_data_16_complete.csv"
balance_file <- "data/processed/complete_case_outcome_balance.csv"

model_data <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = TRUE
)

# 2. Define the same 16 predictors used in the modelling table
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

# 3. Keep only participants with no missing values
#    across the 16 predictors
complete_data <- model_data %>%
  drop_na(all_of(selected_predictors))

# 4. Check the final train/test and outcome counts
outcome_balance <- complete_data %>%
  count(training, PCRpositive, name = "participants") %>%
  group_by(training) %>%
  mutate(percent = round(100 * participants / sum(participants), 1)) %>%
  ungroup()

# 5. Save the complete-case dataset and the class-balance report
write.csv(
  complete_data,
  output_file,
  row.names = FALSE
)

write.csv(
  outcome_balance,
  balance_file,
  row.names = FALSE
)

# 6. Print a concise summary
cat("\nComplete-case dataset created.\n")
cat("Original participants:", nrow(model_data), "\n")
cat("Complete participants:", nrow(complete_data), "\n")
cat("Removed for missing data:", nrow(model_data) - nrow(complete_data), "\n\n")

print(outcome_balance)

cat("\nSaved:", output_file, "\n")
