# prepare_influenza_data.R
# Clean the paper's prepared CSV and create model-ready datasets.

library(dplyr)
library(tidyr)

# 1. File locations
input_file <- "data/raw/Mettelman_Minimum_dataframe.csv"
output_dir <- "data/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# 2. Read data
data <- read.csv(
  input_file,
  encoding = "utf-8",
  stringsAsFactors = FALSE,
  check.names = TRUE
)

# 3. Replace Excel error strings with proper missing values
data[] <- lapply(data, function(x) {
  if (is.character(x)) {
    x[x == "#XL_EVAL_ERROR#"] <- NA_character_
  }
  x
})

# 4. Remove the six participant outliers excluded in the paper
outlier_ids <- c(
  "wn007300", "wn904813", "wn000616",
  "wn006622", "wn000061", "wn904633"
)

data <- data %>%
  filter(!Study.ID %in% outlier_ids)

# Warn if participant IDs are duplicated
if (anyDuplicated(data$Study.ID) > 0) {
  warning("Duplicate Study.ID values were found.")
}

# 5. Define predictor groups
base_predictors <- c(
  "Age_Group", "Sex", "BMI_Cat", "Ethnicity", "Flu_Vaccine_2018",
  "hai_h1", "hai_h3", "hai_bVic", "hai_bYam",
  "nai_h1", "nai_h3", "nai_bVic", "nai_bYam",
  "AUC_H1", "AUC_H3", "AUC_N1", "AUC_N2",
  "AUC_VHA", "AUC_VNA", "AUC_YHA", "AUC_YNA"
)

lymphoid_predictors <- c(
  "CD4.Tcells", "CD4.TNFa", "CD4.Effector", "CD4.IL2",
  "CD4.Naive", "CD4.PD1", "CD4.Th17", "CD4.DP",
  "CD8.Tcells", "CD8.CD107A", "CD8.Effector", "CD8.IFNg",
  "CD8.IL2", "CD8.Memory.CCR5p", "CD8.PD1", "CD8.TNFa",
  "CD8.SP", "CD8.DP", "cTfh.CXCR3p", "cTfh.ICOSp",
  "Gamma.Delta", "NK.GZBn.IFNp", "NK.GZBp.IFNp", "NK.GZBp.IFNn"
)

myeloid_predictors <- c(
  "mDC", "NK.CK.Producing", "NK.Activated", "NK.Cytotoxic",
  "Monocytes.Intermediate", "Monocytes.Nonclassical",
  "Basophils", "Eosinophils"
)

base_vars <- c("PCRpositive", base_predictors)
lymphoid_vars <- c(base_vars, lymphoid_predictors)
myeloid_vars <- c(base_vars, myeloid_predictors)
combined_vars <- union(lymphoid_vars, myeloid_vars)

# 6. Check that required columns exist
required_columns <- unique(c("Study.ID", "training", combined_vars))
missing_columns <- setdiff(required_columns, names(data))

if (length(missing_columns) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

# 7. Convert variables to the correct types
categorical_columns <- c(
  "Age_Group", "Sex", "BMI_Cat",
  "Ethnicity", "Flu_Vaccine_2018", "training"
)

numeric_columns <- setdiff(
  unique(c(base_predictors, lymphoid_predictors, myeloid_predictors)),
  categorical_columns
)

data <- data %>%
  mutate(across(all_of(categorical_columns), as.factor)) %>%
  mutate(
    across(
      all_of(numeric_columns),
      ~ suppressWarnings(as.numeric(as.character(.x)))
    )
  ) %>%
  mutate(
    PCRpositive = factor(
      PCRpositive,
      levels = c(1, 0),
      labels = c("Positive", "Negative")
    )
  )

# 8. Use the train/test labels already supplied in the CSV
split_values <- unique(as.character(data$training))

if (!all(c("training", "testing") %in% split_values)) {
  stop("The 'training' column must contain both 'training' and 'testing'.")
}

training_data <- data %>%
  filter(training == "training")

testing_data <- data %>%
  filter(training == "testing")

# 9. Remove rows with missing values separately for each model
make_complete_dataset <- function(df, variables) {
  df %>%
    select(all_of(variables)) %>%
    drop_na()
}

model_data <- list(
  training_base = make_complete_dataset(training_data, base_vars),
  testing_base = make_complete_dataset(testing_data, base_vars),
  training_lymphoid = make_complete_dataset(training_data, lymphoid_vars),
  testing_lymphoid = make_complete_dataset(testing_data, lymphoid_vars),
  training_myeloid = make_complete_dataset(training_data, myeloid_vars),
  testing_myeloid = make_complete_dataset(testing_data, myeloid_vars),
  training_combined = make_complete_dataset(training_data, combined_vars),
  testing_combined = make_complete_dataset(testing_data, combined_vars)
)

# 10. Create simple quality-control summaries
missing_report <- data.frame(
  variable = names(data),
  missing_n = sapply(data, function(x) sum(is.na(x))),
  missing_percent = round(
    100 * sapply(data, function(x) mean(is.na(x))),
    2
  )
)

dataset_sizes <- data.frame(
  dataset = names(model_data),
  participants = sapply(model_data, nrow),
  variables = sapply(model_data, ncol)
)

# 11. Save outputs
write.csv(
  data,
  file.path(output_dir, "Mettelman_cleaned.csv"),
  row.names = FALSE
)

write.csv(
  missing_report,
  file.path(output_dir, "missing_value_report.csv"),
  row.names = FALSE
)

write.csv(
  dataset_sizes,
  file.path(output_dir, "model_dataset_sizes.csv"),
  row.names = FALSE
)

saveRDS(
  model_data,
  file.path(output_dir, "model_ready_data.rds")
)

print(dataset_sizes)
message("Finished. Prepared files are in: ", output_dir)
