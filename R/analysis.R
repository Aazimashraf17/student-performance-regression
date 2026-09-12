# Student performance regression in base R. No extra R packages are needed.
# Terminal (from repository root): Rscript R/analysis.R
# RStudio: open student-performance-regression.Rproj, then source("R/analysis.R").

run_analysis <- function(root = ".") {
  source_file <- file.path(root, "data", "student-mat.csv")
  if (unname(tools::md5sum(source_file)) != "4dc304be95c60de6ee13fb8769469dd7") {
    stop("Dataset changed: restore the bundled student-mat.csv before using its split.")
  }
  students <- read.csv(source_file, sep = ";", stringsAsFactors = FALSE)
  split <- read.csv(file.path(root, "data", "split.csv"), stringsAsFactors = FALSE)
  stopifnot(identical(names(split), c("row_id", "split")))
  split <- split[order(split$row_id), ]
  stopifnot(identical(as.integer(split$row_id), seq_len(nrow(students))))
  stopifnot(sum(split$split == "train") == 316, sum(split$split == "test") == 79)
  stopifnot(all(split$split %in% c("train", "test")))
  grades <- students[c("G1", "G2", "G3")]
  stopifnot(!anyNA(grades), all(as.matrix(grades) >= 0), all(as.matrix(grades) <= 20))
  students$row_id <- seq_len(nrow(students))
  students$split <- split$split
  train <- students[students$split == "train", ]
  test <- students[students$split == "test", ]

  # lm() fits ordinary least squares. The intercept-only fit predicts the train mean.
  models <- list(
    mean_baseline = lm(G3 ~ 1, data = train),
    simple_G2 = lm(G3 ~ G2, data = train),
    multiple_G1_G2 = lm(G3 ~ G1 + G2, data = train)
  )
  metrics <- list()
  predictions <- list()
  coefficients <- list()
  for (name in names(models)) {
    predicted <- as.numeric(predict(models[[name]], newdata = test))
    residual <- test$G3 - predicted
    metrics[[name]] <- data.frame(
      model = name, test_rows = nrow(test),
      mae = mean(abs(residual)), rmse = sqrt(mean(residual^2)),
      r2 = 1 - sum(residual^2) / sum((test$G3 - mean(test$G3))^2)
    )
    predictions[[name]] <- data.frame(
      row_id = test$row_id, model = name,
      actual = test$G3, predicted = predicted, residual = residual
    )
    estimates <- coef(models[[name]])
    terms <- names(estimates)
    terms[terms == "(Intercept)"] <- "intercept"
    coefficients[[name]] <- data.frame(model = name, term = terms, estimate = unname(estimates))
  }
  metric_table <- do.call(rbind, metrics)
  prediction_table <- do.call(rbind, predictions)
  coefficient_table <- do.call(rbind, coefficients)
  output <- file.path(root, "reports", "r")
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  write.csv(metric_table, file.path(output, "metrics.csv"), row.names = FALSE)
  write.csv(prediction_table, file.path(output, "predictions.csv"), row.names = FALSE)
  write.csv(coefficient_table, file.path(output, "coefficients.csv"), row.names = FALSE)
  capture.output(summary(models$multiple_G1_G2), file = file.path(output, "model_summary.txt"))
  capture.output(sessionInfo(), file = file.path(output, "session_info.txt"))

  # Base-R diagnostics describe training residuals, not test prediction accuracy.
  pdf(file.path(output, "training_diagnostics.pdf"), width = 10, height = 8)
  old_par <- par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  tryCatch(plot(models$multiple_G1_G2, id.n = 0), finally = {
    par(old_par)
    dev.off()
  })
  print(metric_table, row.names = FALSE, digits = 5)
  cat("\nR results saved in reports/r. Test scores use raw, unrounded predictions.\n")
  invisible(list(models = models, metrics = metric_table, predictions = prediction_table))
}

results <- run_analysis()
