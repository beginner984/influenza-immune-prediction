# Run this script from the ROOT of the main repository:
#
# source("shiny-app/setup_app_assets.R")
#
# It copies fitted models and selected aggregate results into the
# Shiny app directory and creates safe aggregate predictor ranges.
# No participant-level data are copied into the app.

find_repository_root <- function() {
  candidates <- unique(
    normalizePath(
      c(
        getwd(),
        file.path(getwd(), ".."),
        file.path(getwd(), "../..")
      ),
      winslash = "/",
      mustWork = FALSE
    )
  )

  valid <- candidates[
    file.exists(file.path(candidates, "R")) &
      file.exists(
        file.path(
          candidates,
          "data/processed/training_data_16.csv"
        )
      )
  ]

  if (length(valid) == 0) {
    stop(
      "Could not locate the repository root. ",
      "Run this script from the repository root."
    )
  }

  valid[1]
}

repo_root <- find_repository_root()
app_dir <- file.path(repo_root, "shiny-app")

dir.create(
  file.path(app_dir, "models"),
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  file.path(app_dir, "data"),
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  file.path(app_dir, "results"),
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  file.path(app_dir, "www"),
  recursive = TRUE,
  showWarnings = FALSE
)

copy_required <- function(source, destination) {
  if (!file.exists(source)) {
    stop("Required file not found: ", source)
  }

  success <- file.copy(
    source,
    destination,
    overwrite = TRUE
  )

  if (!success) {
    stop("Could not copy: ", source)
  }
}

copy_optional <- function(source, destination) {
  if (file.exists(source)) {
    file.copy(
      source,
      destination,
      overwrite = TRUE
    )
  }
}

required_model_files <- c(
  "elastic_net_model.rds",
  "random_forest_model.rds",
  "xgboost_native_model.json",
  "xgboost_native_metadata.rds"
)

for (filename in required_model_files) {
  copy_required(
    file.path(repo_root, "models", filename),
    file.path(app_dir, "models", filename)
  )
}

copy_optional(
  file.path(
    repo_root,
    "models/ensemble_isotonic_calibrator.rds"
  ),
  file.path(
    app_dir,
    "models/ensemble_isotonic_calibrator.rds"
  )
)

required_result_files <- c(
  "elastic_net_cv_performance.csv",
  "random_forest_cv_performance.csv",
  "xgboost_native_cv_performance.csv",
  "ensemble_cv_performance.csv",
  "external_test_model_comparison.csv",
  "variable_importance_table_publication.csv"
)

for (filename in required_result_files) {
  copy_required(
    file.path(repo_root, "results", filename),
    file.path(app_dir, "results", filename)
  )
}

figure_files <- c(
  "Figure_2_ROC_test.png",
  "Figure_3_calibration_test.png"
)

for (filename in figure_files) {
  copy_required(
    file.path(repo_root, "results", filename),
    file.path(app_dir, "www", filename)
  )
}

training_data <- read.csv(
  file.path(
    repo_root,
    "data/processed/training_data_16.csv"
  ),
  stringsAsFactors = FALSE,
  check.names = TRUE
)

metadata <- readRDS(
  file.path(
    repo_root,
    "models/xgboost_native_metadata.rds"
  )
)

predictors <- metadata$predictors

missing_predictors <- setdiff(
  predictors,
  names(training_data)
)

if (length(missing_predictors) > 0) {
  stop(
    "Training data are missing predictors: ",
    paste(missing_predictors, collapse = ", ")
  )
}

predictor_ranges <- do.call(
  rbind,
  lapply(predictors, function(variable) {
    values <- as.numeric(training_data[[variable]])

    observed_min <- min(values, na.rm = TRUE)
    observed_max <- max(values, na.rm = TRUE)
    observed_range <- observed_max - observed_min

    step_value <- if (
      is.finite(observed_range) &&
        observed_range > 0
    ) {
      signif(observed_range / 100, 3)
    } else {
      0.01
    }

    data.frame(
      variable = variable,
      minimum = observed_min,
      maximum = observed_max,
      median = median(values, na.rm = TRUE),
      q05 = unname(
        quantile(values, 0.05, na.rm = TRUE)
      ),
      q95 = unname(
        quantile(values, 0.95, na.rm = TRUE)
      ),
      step = step_value,
      stringsAsFactors = FALSE
    )
  })
)

write.csv(
  predictor_ranges,
  file.path(app_dir, "data/predictor_ranges.csv"),
  row.names = FALSE
)

cat("\nShiny assets prepared successfully.\n")
cat("App directory:\n", app_dir, "\n\n")
cat("Launch the app with:\n")
cat('shiny::runApp("shiny-app")\n\n')
cat(
  "No participant-level data were copied into the app.\n"
)
