# Influenza immune-data preparation

A small, interview-ready R project that prepares the Mettelman influenza immunology dataset for statistical and predictive modelling.

The workflow is based on the data-preparation logic in the authors' public analysis code, but rewritten as one clear and reproducible script.

## What the script does

1. Reads the prepared CSV.
2. Converts Excel error strings to proper missing values.
3. Removes the six participant outliers excluded in the paper.
4. Checks participant IDs and required columns.
5. Converts categorical and continuous variables to suitable R types.
6. Defines Base, Lymphoid, Myeloid and Combined predictor sets.
7. Uses the existing training/testing labels in the supplied dataframe.
8. Applies complete-case filtering separately for each model.
9. Saves cleaned data, missingness summaries and model-ready datasets.

## Repository structure

```text
influenza-immune-prediction/
├── R/
│   └── prepare_influenza_data.R
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── .gitignore
└── README.md
```

## How to run

Install the required packages once:

```r
install.packages(c("dplyr", "tidyr"))
```

Place the source dataframe here:

```text
data/raw/Mettelman_Minimum_dataframe.csv
```

From the repository root, run:

```r
source("R/prepare_influenza_data.R")
```

The script writes the following files to `data/processed/`:

- `Mettelman_cleaned.csv`
- `missing_value_report.csv`
- `model_dataset_sizes.csv`
- `model_ready_data.rds`

## Important methodological note

The authors' code begins with an already prepared minimum dataframe. Therefore, this repository does not reproduce raw flow-cytometry gating, laboratory-assay processing, or creation of the original train/test allocation. It prepares the supplied analysis dataframe for downstream modelling.

## Suggested next steps

Future analysis scripts can be added under `R/`, for example:

```text
R/01_prepare_data.R
R/02_descriptive_analysis.R
R/03_logistic_regression.R
R/04_random_forest.R
```
