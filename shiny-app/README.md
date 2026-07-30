# Influenza Immune Risk Explorer

A local R Shiny application for exploring predictions from the fitted:

- Elastic Net model
- Random Forest model
- XGBoost model
- Equal-weight ensemble

The app contains:

1. **Outcome Prediction**
2. **Individual Contributions**
3. **Model Performance**
4. **About & Disclaimer**

## Important

This is an exploratory research prototype, not a clinical application.

The app does not retrain models and does not save user-entered immune values.

## 1. Install packages

From R:

```r
install.packages(c(
  "shiny",
  "bslib",
  "ggplot2",
  "dplyr",
  "caret",
  "glmnet",
  "ranger",
  "xgboost",
  "scales"
))
```

## 2. Add this directory to the repository

Copy the complete `shiny-app` folder into:

```text
influenza-immune-prediction/
```

The structure should look like:

```text
influenza-immune-prediction/
├── R/
├── data/
├── models/
├── results/
└── shiny-app/
```

## 3. Prepare app assets

From the root of the main repository, run:

```r
source("shiny-app/setup_app_assets.R")
```

This copies the fitted models and selected aggregate results into the app.

It also creates `predictor_ranges.csv` using summary statistics from the training set. It does **not** copy participant-level data.

## 4. Launch the app

From the root of the main repository:

```r
shiny::runApp("shiny-app")
```

Alternatively:

```r
setwd("shiny-app")
shiny::runApp()
```

## Prediction logic

The app calculates:

```text
ensemble probability =
(Elastic Net + Random Forest + XGBoost) / 3
```

The fixed ensemble decision threshold is read from:

```text
results/ensemble_cv_performance.csv
```

## Individual contribution plot

For each predictor, the app:

1. calculates the current ensemble probability;
2. replaces that predictor with its training-set median;
3. recalculates the ensemble probability;
4. reports the probability difference.

This is a one-at-a-time counterfactual explanation. It is **not** a SHAP value, is not causal, and contributions do not necessarily sum to the total prediction.

## Deployment

Test the app locally first.

For deployment to shinyapps.io:

```r
install.packages("rsconnect")
rsconnect::setAccountInfo(
  name = "YOUR_ACCOUNT",
  token = "YOUR_TOKEN",
  secret = "YOUR_SECRET"
)

rsconnect::deployApp("shiny-app")
```

Do not place raw or processed participant-level data inside the deployment directory.

## Git privacy

The model files are needed to run and deploy the app, but they do not need to be committed to the public analysis repository.

Add this to the repository `.gitignore`:

```text
/shiny-app/models/*
!/shiny-app/models/.gitkeep
/shiny-app/data/predictor_ranges.csv
```

The app source code can still be committed.
