# R/15_prepare_vaccination_sensitivity.R
# Prepare the post-hoc sensitivity-analysis datasets containing:
#   16 immune predictors + vaccination status
#
# This does not train any models.
# It preserves the existing training/test split.

library(dplyr)

input_file <- "data/processed/modelling_data_16_plus_vaccination.csv"

training_output <- paste0(
  "data/processed/",
  "training_data_16_plus_vaccination.csv"
)

testing_output <- paste0(
  "data/processed/",
  "testing_data_16_plus_vaccination.csv"
)

# ------------------------------------------------------------
# 1. Read the 171-participant sensitivity dataset
# ------------------------------------------------------------

sensitivity_data <- read.csv(
  input_file,
  stringsAsFactors = FALSE,
  check.names = TRUE
)

required_columns <- c(
  "Study.ID",
  "training",
  "PCRpositive",
  "Flu_Vaccine_2018"
)

missing_columns <- setdiff(
  required_columns,
  names(sensitivity_data)
)

if (length(missing_columns) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

# ------------------------------------------------------------
# 2. Standardise factor levels
# ------------------------------------------------------------

sensitivity_data$PCRpositive <- factor(
  sensitivity_data$PCRpositive,
  levels = c("Positive", "Negative")
)

sensitivity_data$Flu_Vaccine_2018 <- factor(
  sensitivity_data$Flu_Vaccine_2018,
  levels = c("Unvaccinated", "Vaccinated")
)

if (any(is.na(sensitivity_data$Flu_Vaccine_2018))) {
  stop("Vaccination status contains missing or unexpected values.")
}

# ------------------------------------------------------------
# 3. Preserve the original fixed split
# ------------------------------------------------------------

training_data <- sensitivity_data %>%
  filter(training == "training") %>%
  select(-training)

testing_data <- sensitivity_data %>%
  filter(training == "testing") %>%
  select(-training)

# ------------------------------------------------------------
# 4. Validate counts and participant overlap
# ------------------------------------------------------------

if (nrow(training_data) != 134) {
  stop(
    "Expected 134 training participants but found ",
    nrow(training_data),
    "."
  )
}

if (nrow(testing_data) != 37) {
  stop(
    "Expected 37 testing participants but found ",
    nrow(testing_data),
    "."
  )
}

if (
  length(
    intersect(
      training_data$Study.ID,
      testing_data$Study.ID
    )
  ) > 0
) {
  stop("Participant overlap detected between training and testing.")
}

# ------------------------------------------------------------
# 5. Save analysis files
# ------------------------------------------------------------

write.csv(
  training_data,
  training_output,
  row.names = FALSE
)

write.csv(
  testing_data,
  testing_output,
  row.names = FALSE
)

# ------------------------------------------------------------
# 6. Print summary
# ------------------------------------------------------------

cat("\nVaccination sensitivity datasets prepared.\n\n")

cat("Training participants:", nrow(training_data), "\n")
cat("Testing participants:", nrow(testing_data), "\n\n")

cat("Training distribution:\n")
print(
  table(
    Vaccination = training_data$Flu_Vaccine_2018,
    Outcome = training_data$PCRpositive
  )
)

cat("\nTesting distribution:\n")
print(
  table(
    Vaccination = testing_data$Flu_Vaccine_2018,
    Outcome = testing_data$PCRpositive
  )
)

cat("\nTraining symptomatic proportion by vaccination:\n")
print(
  prop.table(
    table(
      Vaccination = training_data$Flu_Vaccine_2018,
      Outcome = training_data$PCRpositive
    ),
    margin = 1
  )[, "Positive"]
)

cat("\nTesting symptomatic proportion by vaccination:\n")
print(
  prop.table(
    table(
      Vaccination = testing_data$Flu_Vaccine_2018,
      Outcome = testing_data$PCRpositive
    ),
    margin = 1
  )[, "Positive"]
)

cat("\nSaved:\n")
cat(training_output, "\n")
cat(testing_output, "\n")

cat(
  "\nThis is a post-hoc sensitivity analysis because ",
  "the original held-out test results have already been viewed.\n",
  sep = ""
)
