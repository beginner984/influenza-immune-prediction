library(shiny)
library(ggplot2)
library(dplyr)
library(caret)
library(xgboost)
library(bslib)
library(scales)

source("R/prediction_helpers.R", local = TRUE)

required_files <- c(
  "models/elastic_net_model.rds",
  "models/random_forest_model.rds",
  "models/xgboost_native_model.json",
  "models/xgboost_native_metadata.rds",
  "data/predictor_ranges.csv",
  "results/elastic_net_cv_performance.csv",
  "results/random_forest_cv_performance.csv",
  "results/xgboost_native_cv_performance.csv",
  "results/ensemble_cv_performance.csv",
  "results/external_test_model_comparison.csv",
  "results/variable_importance_table_publication.csv"
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop(
    paste0(
      "\nThe Shiny app assets have not been prepared.\n\n",
      "Missing files:\n- ",
      paste(missing_files, collapse = "\n- "),
      "\n\nFrom the repository root, run:\n",
      'source("shiny-app/setup_app_assets.R")\n'
    ),
    call. = FALSE
  )
}

elastic_model <- readRDS("models/elastic_net_model.rds")
rf_model <- readRDS("models/random_forest_model.rds")
xgb_model <- xgb.load("models/xgboost_native_model.json")
xgb_metadata <- readRDS("models/xgboost_native_metadata.rds")

isotonic_calibrator <- if (
  file.exists("models/ensemble_isotonic_calibrator.rds")
) {
  readRDS("models/ensemble_isotonic_calibrator.rds")
} else {
  NULL
}

predictor_ranges <- read.csv(
  "data/predictor_ranges.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

predictor_names <- xgb_metadata$predictors

if (!all(predictor_names %in% predictor_ranges$variable)) {
  stop("predictor_ranges.csv does not contain every model predictor.")
}

predictor_ranges <- predictor_ranges[
  match(predictor_names, predictor_ranges$variable),
]

thresholds <- c(
  Elastic_Net = read.csv(
    "results/elastic_net_cv_performance.csv"
  )$threshold[1],
  Random_Forest = read.csv(
    "results/random_forest_cv_performance.csv"
  )$threshold[1],
  XGBoost = read.csv(
    "results/xgboost_native_cv_performance.csv"
  )$threshold[1],
  Ensemble = read.csv(
    "results/ensemble_cv_performance.csv"
  )$threshold[1]
)

test_performance <- read.csv(
  "results/external_test_model_comparison.csv",
  stringsAsFactors = FALSE
)

global_importance <- read.csv(
  "results/variable_importance_table_publication.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

group_map <- list(
  "cTfh markers" = c(
    "cTfh.ICOSp",
    "cTfh.CXCR3p"
  ),
  "CD4 markers" = c(
    "CD4.Effector",
    "CD4.IL2",
    "CD4.Naive",
    "CD4.DP"
  ),
  "CD8 markers" = c(
    "CD8.Effector",
    "CD8.Memory.CCR5p",
    "CD8.SP",
    "CD8.DP",
    "CD8.Tcells"
  ),
  "NK and other markers" = c(
    "NK.Activated",
    "NK.GZBp.IFNp",
    "NK.CK.Producing",
    "Gamma.Delta",
    "nai_h3"
  )
)

input_id_for <- function(variable) {
  paste0("feature_", make.names(variable))
}

theme <- bs_theme(
  version = 5,
  bg = "#f5f7fb",
  fg = "#172033",
  primary = "#3767e8",
  secondary = "#6b7280",
  success = "#18a875",
  danger = "#d9534f",
  base_font = font_google("Inter"),
  heading_font = font_google("Inter")
)

ui <- fluidPage(
  theme = theme,

  tags$head(
    tags$link(
      rel = "stylesheet",
      type = "text/css",
      href = "style.css"
    ),
    tags$meta(
      name = "viewport",
      content = "width=device-width, initial-scale=1"
    )
  ),

  div(
    class = "app-hero",
    div(
      class = "hero-copy",
      tags$span(class = "hero-kicker", "RESEARCH PROTOTYPE"),
      tags$h1("Influenza Immune Risk Explorer"),
      tags$p(
        "Explore the predicted probability of symptomatic influenza ",
        "from 16 baseline immune-profile measurements."
      )
    ),
    div(
      class = "hero-badge",
      tags$span("171 participants"),
      tags$span("16 predictors"),
      tags$span("4 models")
    )
  ),

  div(
    class = "app-shell",

    div(
      class = "input-sidebar",

      div(
        class = "sidebar-heading",
        tags$h3("Immune profile"),
        tags$p(
          "Enter values or keep the training-set median profile."
        )
      ),

      uiOutput("predictor_controls"),

      div(
        class = "sidebar-actions",
        actionButton(
          "reset_inputs",
          "Reset to medians",
          class = "btn-reset"
        ),
        downloadButton(
          "download_prediction",
          "Download prediction",
          class = "btn-download"
        )
      ),

      div(
        class = "privacy-note",
        tags$strong("Privacy"),
        tags$p(
          "Values are used only in the current browser session ",
          "and are not written to disk."
        )
      )
    ),

    div(
      class = "main-stage",

      tabsetPanel(
        id = "main_tabs",
        type = "tabs",

        tabPanel(
          title = "Outcome Prediction",
          value = "prediction",

          div(
            class = "metrics-grid",
            div(
              class = "metric-card primary-metric",
              tags$span(class = "metric-label", "Ensemble probability"),
              uiOutput("ensemble_probability_text"),
              tags$span(
                class = "metric-footnote",
                "Primary unscaled probability"
              )
            ),
            div(
              class = "metric-card",
              tags$span(class = "metric-label", "Predicted class"),
              uiOutput("predicted_class_text"),
              tags$span(
                class = "metric-footnote",
                paste0(
                  "Fixed threshold: ",
                  sprintf("%.3f", thresholds["Ensemble"])
                )
              )
            ),
            div(
              class = "metric-card",
              tags$span(class = "metric-label", "Model agreement"),
              uiOutput("agreement_text"),
              tags$span(
                class = "metric-footnote",
                "Models above their own fixed thresholds"
              )
            )
          ),

          div(
            class = "content-grid two-column",

            div(
              class = "panel-card gauge-card",
              tags$div(
                class = "panel-heading",
                tags$h3("Probability of symptomatic influenza"),
                tags$p(
                  "The prediction updates when an immune value changes."
                )
              ),
              uiOutput("probability_gauge"),
              uiOutput("calibrated_probability_note")
            ),

            div(
              class = "panel-card",
              tags$div(
                class = "panel-heading",
                tags$h3("Component-model probabilities"),
                tags$p(
                  "Elastic Net, Random Forest and XGBoost are ",
                  "averaged equally."
                )
              ),
              plotOutput(
                "model_probability_plot",
                height = "330px"
              )
            )
          ),

          div(
            class = "panel-card",
            tags$div(
              class = "panel-heading",
              tags$h3("How to read this result"),
              tags$p(
                "This is an exploratory model output, not a diagnosis."
              )
            ),
            uiOutput("interpretation_text")
          )
        ),

        tabPanel(
          title = "Individual Contributions",
          value = "contributions",

          div(
            class = "section-intro",
            tags$h2("Which inputs changed this prediction?"),
            tags$p(
              "Each contribution compares the current value with the ",
              "training-set median while holding all other values fixed."
            )
          ),

          div(
            class = "content-grid two-column",

            div(
              class = "panel-card",
              tags$div(
                class = "panel-heading",
                tags$h3("Current-profile contributions"),
                tags$p(
                  "Positive values increase the predicted probability; ",
                  "negative values decrease it."
                )
              ),
              plotOutput(
                "local_contribution_plot",
                height = "480px"
              ),
              tags$p(
                class = "figure-note",
                "These one-at-a-time effects are not SHAP values and ",
                "do not necessarily add to the total prediction."
              )
            ),

            div(
              class = "panel-card",
              tags$div(
                class = "panel-heading",
                tags$h3("Global ensemble importance"),
                tags$p(
                  "Average normalized importance across the three ",
                  "component models."
                )
              ),
              plotOutput(
                "global_importance_plot",
                height = "480px"
              ),
              tags$p(
                class = "figure-note",
                "Global importance describes prediction, not causation."
              )
            )
          )
        ),

        tabPanel(
          title = "Model Performance",
          value = "performance",

          div(
            class = "section-intro",
            tags$h2("Held-out test performance"),
            tags$p(
              "The test set contained 37 participants, including ",
              "8 symptomatic influenza cases."
            )
          ),

          div(
            class = "panel-card",
            tags$div(
              class = "panel-heading",
              tags$h3("Performance summary"),
              tags$p(
                "Thresholds were selected using out-of-fold ",
                "training predictions only."
              )
            ),
            tableOutput("performance_table")
          ),

          div(
            class = "content-grid two-column",
            div(
              class = "panel-card image-panel",
              tags$h3("ROC curves"),
              conditionalPanel(
                condition = "true",
                uiOutput("roc_image")
              )
            ),
            div(
              class = "panel-card image-panel",
              tags$h3("Calibration"),
              uiOutput("calibration_image")
            )
          )
        ),

        tabPanel(
          title = "About & Disclaimer",
          value = "about",

          div(
            class = "about-grid",

            div(
              class = "panel-card",
              tags$h2("What this app does"),
              tags$p(
                "The app applies previously fitted Elastic Net, ",
                "Random Forest and XGBoost models to a user-entered ",
                "immune profile. The ensemble is the equal-weight ",
                "average of the three probabilities."
              ),
              tags$ul(
                tags$li(
                  "Predictors and thresholds are fixed."
                ),
                tags$li(
                  "No model is retrained inside the app."
                ),
                tags$li(
                  "Inputs are not saved."
                ),
                tags$li(
                  "The unscaled ensemble probability is the primary output."
                )
              )
            ),

            div(
              class = "panel-card warning-card",
              tags$h2("Research-use disclaimer"),
              tags$p(
                "This application is an exploratory research prototype. ",
                "It has not been validated for clinical diagnosis, ",
                "treatment selection, patient counselling or public-health ",
                "decision-making."
              ),
              tags$p(
                "The held-out test set was small and contained only ",
                "eight symptomatic cases. Predictions may be uncertain ",
                "or unreliable outside the source study population."
              )
            ),

            div(
              class = "panel-card",
              tags$h2("Model limitations"),
              tags$ul(
                tags$li("Complete-case analysis was used."),
                tags$li(
                  "The predictor list was informed by the source study."
                ),
                tags$li(
                  "There is no independent external validation cohort."
                ),
                tags$li(
                  "Variable importance should not be interpreted causally."
                )
              )
            )
          )
        )
      )
    )
  ),

  tags$footer(
    class = "app-footer",
    "Influenza Immune Risk Explorer · Research prototype"
  )
)

server <- function(input, output, session) {

  output$predictor_controls <- renderUI({
    tagList(
      lapply(seq_along(group_map), function(group_index) {
        group_name <- names(group_map)[group_index]
        variables <- group_map[[group_index]]
        variables <- variables[variables %in% predictor_names]

        tags$details(
          class = "feature-group",
          open = if (group_index == 1) NA else NULL,
          tags$summary(group_name),
          lapply(variables, function(variable) {
            row <- predictor_ranges[
              predictor_ranges$variable == variable,
            ]

            sliderInput(
              inputId = input_id_for(variable),
              label = variable,
              min = row$minimum,
              max = row$maximum,
              value = row$median,
              step = row$step,
              ticks = FALSE
            )
          })
        )
      })
    )
  })

  observeEvent(input$reset_inputs, {
    for (variable in predictor_names) {
      row <- predictor_ranges[
        predictor_ranges$variable == variable,
      ]

      updateSliderInput(
        session,
        inputId = input_id_for(variable),
        value = row$median
      )
    }
  })

  current_profile <- reactive({
    values <- vapply(
      predictor_names,
      function(variable) {
        value <- input[[input_id_for(variable)]]

        if (is.null(value) || !is.finite(value)) {
          predictor_ranges$median[
            predictor_ranges$variable == variable
          ]
        } else {
          as.numeric(value)
        }
      },
      numeric(1)
    )

    profile <- as.data.frame(
      as.list(values),
      check.names = FALSE
    )

    names(profile) <- predictor_names
    profile
  })

  model_predictions <- reactive({
    predict_all_models(
      profile = current_profile(),
      elastic_model = elastic_model,
      rf_model = rf_model,
      xgb_model = xgb_model,
      xgb_predictors = predictor_names
    )
  })

  model_classes <- reactive({
    probabilities <- model_predictions()

    c(
      Elastic_Net = unname(
        probabilities["Elastic Net"] >= thresholds["Elastic_Net"]
      ),
      Random_Forest = unname(
        probabilities["Random Forest"] >= thresholds["Random_Forest"]
      ),
      XGBoost = unname(
        probabilities["XGBoost"] >= thresholds["XGBoost"]
      ),
      Ensemble = unname(
        probabilities["Equal-weight ensemble"] >= thresholds["Ensemble"]
      )
    )
  })

  output$ensemble_probability_text <- renderUI({
    probability <- model_predictions()["Equal-weight ensemble"]

    tags$span(
      class = "metric-value",
      percent(probability, accuracy = 0.1)
    )
  })

  output$predicted_class_text <- renderUI({
    positive <- model_classes()["Ensemble"]

    tags$span(
      class = paste(
        "class-pill",
        if (positive) "class-positive" else "class-negative"
      ),
      if (positive) "Symptomatic" else "Protected"
    )
  })

  output$agreement_text <- renderUI({
    number_positive <- sum(
      model_classes()[c(
        "Elastic_Net",
        "Random_Forest",
        "XGBoost"
      )]
    )

    tags$span(
      class = "metric-value",
      paste0(number_positive, " of 3")
    )
  })

  output$probability_gauge <- renderUI({
    probability <- unname(
      model_predictions()["Equal-weight ensemble"]
    )

    degrees <- 360 * probability

    div(
      class = "gauge-wrap",
      div(
        class = "probability-gauge",
        style = paste0(
          "--gauge-degrees:",
          sprintf("%.2fdeg", degrees),
          ";"
        ),
        div(
          class = "gauge-inner",
          tags$span(
            class = "gauge-number",
            percent(probability, accuracy = 0.1)
          ),
          tags$span(
            class = "gauge-caption",
            "predicted probability"
          )
        )
      ),
      div(
        class = "gauge-scale",
        tags$span("0%"),
        tags$span(
          class = "threshold-marker",
          paste0(
            "Threshold ",
            percent(
              thresholds["Ensemble"],
              accuracy = 0.1
            )
          )
        ),
        tags$span("100%")
      )
    )
  })

  output$calibrated_probability_note <- renderUI({
    if (is.null(isotonic_calibrator)) {
      return(NULL)
    }

    raw_probability <- unname(
      model_predictions()["Equal-weight ensemble"]
    )

    calibrated <- apply_isotonic_mapping(
      isotonic_calibrator,
      raw_probability
    )

    div(
      class = "calibration-note",
      tags$strong("Exploratory isotonic estimate: "),
      percent(calibrated, accuracy = 0.1),
      tags$span(
        " — shown for sensitivity analysis only."
      )
    )
  })

  output$model_probability_plot <- renderPlot({
    probabilities <- model_predictions()

    plot_data <- data.frame(
      model = factor(
        names(probabilities),
        levels = rev(names(probabilities))
      ),
      probability = as.numeric(probabilities)
    )

    ggplot(
      plot_data,
      aes(x = probability, y = model)
    ) +
      geom_col(
        width = 0.62,
        fill = "#4f6ee8"
      ) +
      geom_text(
        aes(
          label = percent(
            probability,
            accuracy = 0.1
          )
        ),
        hjust = -0.12,
        size = 4
      ) +
      scale_x_continuous(
        limits = c(0, 1.05),
        breaks = seq(0, 1, by = 0.2),
        labels = percent_format(accuracy = 1),
        expand = expansion(mult = c(0, 0.01))
      ) +
      labs(
        x = "Predicted probability",
        y = NULL
      ) +
      theme_minimal(base_size = 13) +
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.y = element_text(
          colour = "#172033",
          face = "bold"
        ),
        axis.title.x = element_text(
          colour = "#5c667a"
        ),
        plot.margin = margin(10, 35, 10, 10)
      )
  })

  output$interpretation_text <- renderUI({
    probability <- unname(
      model_predictions()["Equal-weight ensemble"]
    )
    positive <- probability >= thresholds["Ensemble"]

    if (positive) {
      div(
        class = "interpretation-box interpretation-positive",
        tags$strong(
          "The probability is above the fixed ensemble threshold."
        ),
        tags$p(
          "The model classifies this profile as symptomatic influenza. ",
          "This classification is a research output and must not be ",
          "used as a clinical diagnosis."
        )
      )
    } else {
      div(
        class = "interpretation-box interpretation-negative",
        tags$strong(
          "The probability is below the fixed ensemble threshold."
        ),
        tags$p(
          "The model classifies this profile as protected. ",
          "A low model probability does not rule out infection or illness."
        )
      )
    }
  })

  local_contributions <- reactive({
    calculate_one_at_a_time_contributions(
      current_profile = current_profile(),
      reference_ranges = predictor_ranges,
      elastic_model = elastic_model,
      rf_model = rf_model,
      xgb_model = xgb_model,
      xgb_predictors = predictor_names
    )
  })

  output$local_contribution_plot <- renderPlot({
    contribution_data <- local_contributions() %>%
      arrange(desc(abs(contribution))) %>%
      slice_head(n = 12) %>%
      arrange(contribution) %>%
      mutate(
        variable = factor(
          variable,
          levels = variable
        ),
        direction = ifelse(
          contribution >= 0,
          "Increases probability",
          "Decreases probability"
        )
      )

    ggplot(
      contribution_data,
      aes(
        x = contribution,
        y = variable,
        fill = direction
      )
    ) +
      geom_col(width = 0.68) +
      geom_vline(
        xintercept = 0,
        colour = "#aab1bf"
      ) +
      scale_fill_manual(
        values = c(
          "Increases probability" = "#e35d6a",
          "Decreases probability" = "#2ca58d"
        )
      ) +
      scale_x_continuous(
        labels = label_percent(
          accuracy = 0.1
        )
      ) +
      labs(
        x = "Change in ensemble probability",
        y = NULL,
        fill = NULL
      ) +
      theme_minimal(base_size = 12) +
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "bottom",
        axis.text.y = element_text(
          colour = "#172033"
        )
      )
  })

  output$global_importance_plot <- renderPlot({
    ensemble_column <- intersect(
      c(
        "Equal_weight_ensemble",
        "Equal.weight.ensemble",
        "Equal-weight ensemble"
      ),
      names(global_importance)
    )[1]

    validate(
      need(
        !is.na(ensemble_column),
        "Ensemble importance column not found."
      )
    )

    importance_data <- global_importance %>%
      transmute(
        variable = variable,
        importance = .data[[ensemble_column]]
      ) %>%
      arrange(desc(importance)) %>%
      slice_head(n = 12) %>%
      arrange(importance) %>%
      mutate(
        variable = factor(
          variable,
          levels = variable
        )
      )

    ggplot(
      importance_data,
      aes(x = importance, y = variable)
    ) +
      geom_col(
        width = 0.68,
        fill = "#4f6ee8"
      ) +
      geom_text(
        aes(
          label = paste0(
            sprintf("%.1f", importance),
            "%"
          )
        ),
        hjust = -0.12,
        size = 3.7
      ) +
      scale_x_continuous(
        limits = c(
          0,
          max(importance_data$importance) * 1.18
        ),
        labels = function(x) paste0(x, "%"),
        expand = expansion(mult = c(0, 0.02))
      ) +
      labs(
        x = "Normalized importance",
        y = NULL
      ) +
      theme_minimal(base_size = 12) +
      theme(
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.y = element_text(
          colour = "#172033"
        )
      )
  })

  output$performance_table <- renderTable({
    test_performance %>%
      transmute(
        Model = model,
        `ROC AUC` = round(ROC_AUC, 3),
        Sensitivity = percent(
          sensitivity,
          accuracy = 0.1
        ),
        Specificity = percent(
          specificity,
          accuracy = 0.1
        ),
        `Balanced accuracy` = percent(
          balanced_accuracy,
          accuracy = 0.1
        ),
        Accuracy = percent(
          accuracy,
          accuracy = 0.1
        )
      )
  }, striped = TRUE, bordered = FALSE, hover = TRUE)

  output$roc_image <- renderUI({
    image_path <- "Figure_2_ROC_test.png"

    if (!file.exists(file.path("www", image_path))) {
      return(
        div(
          class = "missing-image",
          "ROC figure is not available."
        )
      )
    }

    tags$img(
      src = image_path,
      class = "result-image",
      alt = "Held-out test ROC curves for four models"
    )
  })

  output$calibration_image <- renderUI({
    image_path <- "Figure_3_calibration_test.png"

    if (!file.exists(file.path("www", image_path))) {
      return(
        div(
          class = "missing-image",
          "Calibration figure is not available."
        )
      )
    }

    tags$img(
      src = image_path,
      class = "result-image",
      alt = "Ensemble calibration before and after scaling"
    )
  })

  output$download_prediction <- downloadHandler(
    filename = function() {
      paste0(
        "influenza_prediction_",
        format(Sys.Date(), "%Y%m%d"),
        ".csv"
      )
    },
    content = function(file) {
      profile <- current_profile()
      probabilities <- model_predictions()

      output_data <- cbind(
        profile,
        elastic_net_probability =
          probabilities["Elastic Net"],
        random_forest_probability =
          probabilities["Random Forest"],
        xgboost_probability =
          probabilities["XGBoost"],
        ensemble_probability =
          probabilities["Equal-weight ensemble"],
        ensemble_threshold =
          thresholds["Ensemble"],
        predicted_class = ifelse(
          probabilities["Equal-weight ensemble"] >=
            thresholds["Ensemble"],
          "Symptomatic",
          "Protected"
        )
      )

      write.csv(
        output_data,
        file,
        row.names = FALSE
      )
    }
  )
}

shinyApp(ui, server)
