predict_all_models <- function(
  profile,
  elastic_model,
  rf_model,
  xgb_model,
  xgb_predictors
) {
  profile <- as.data.frame(
    profile,
    check.names = FALSE
  )

  elastic_probability <- predict(
    elastic_model,
    newdata = profile,
    type = "prob"
  )[, "Positive"]

  rf_probability <- predict(
    rf_model,
    newdata = profile,
    type = "prob"
  )[, "Positive"]

  xgb_matrix <- as.matrix(
    profile[, xgb_predictors, drop = FALSE]
  )

  xgb_probability <- predict(
    xgb_model,
    xgb_matrix
  )

  ensemble_probability <- mean(
    c(
      elastic_probability,
      rf_probability,
      xgb_probability
    )
  )

  c(
    "Elastic Net" =
      as.numeric(elastic_probability),
    "Random Forest" =
      as.numeric(rf_probability),
    "XGBoost" =
      as.numeric(xgb_probability),
    "Equal-weight ensemble" =
      as.numeric(ensemble_probability)
  )
}

apply_isotonic_mapping <- function(mapping, probability) {
  calibrated <- approx(
    x = mapping$x,
    y = mapping$calibrated,
    xout = as.numeric(probability),
    method = "constant",
    f = 1,
    rule = 2,
    ties = "ordered"
  )$y

  pmin(pmax(calibrated, 0), 1)
}

calculate_one_at_a_time_contributions <- function(
  current_profile,
  reference_ranges,
  elastic_model,
  rf_model,
  xgb_model,
  xgb_predictors
) {
  current_probability <- predict_all_models(
    profile = current_profile,
    elastic_model = elastic_model,
    rf_model = rf_model,
    xgb_model = xgb_model,
    xgb_predictors = xgb_predictors
  )["Equal-weight ensemble"]

  contribution_rows <- lapply(
    xgb_predictors,
    function(variable) {
      counterfactual_profile <- current_profile

      counterfactual_profile[[variable]] <-
        reference_ranges$median[
          reference_ranges$variable == variable
        ]

      counterfactual_probability <- predict_all_models(
        profile = counterfactual_profile,
        elastic_model = elastic_model,
        rf_model = rf_model,
        xgb_model = xgb_model,
        xgb_predictors = xgb_predictors
      )["Equal-weight ensemble"]

      data.frame(
        variable = variable,
        current_value =
          as.numeric(current_profile[[variable]]),
        reference_median =
          reference_ranges$median[
            reference_ranges$variable == variable
          ],
        contribution =
          as.numeric(
            current_probability -
              counterfactual_probability
          )
      )
    }
  )

  dplyr::bind_rows(contribution_rows)
}
