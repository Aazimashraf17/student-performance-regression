# Understand the project

## 1. The question

We know two earlier grades and want to estimate a final grade. The earlier grades are the **features**. The final grade is the **target**. Regression predicts a numerical value.

This is a useful project to discuss in a data science interview because you can explain the full process: inspect data, set aside test records, fit a model, measure its errors, and explain its limits.

## 2. Start with one input

Simple regression fits a straight-line relationship:

```text
y_hat = intercept + slope × x
```

Here x is G2 and y_hat is the predicted G3. The intercept gives the predicted value at x = 0; the slope gives the change in the fitted prediction for a one-unit increase in x. Neither is automatically a causal effect.

Ordinary least squares chooses coefficients that minimize the sum of squared errors in the training records.

## 3. Add another input

Multiple regression uses more than one feature:

```text
predicted G3 = intercept + coefficient_1 × G1 + coefficient_2 × G2
```

Both coefficients are estimated together. Since G1 and G2 are correlated, do not read a coefficient as a standalone importance score. More features do not automatically improve performance on unseen records.

## 4. Keep an honest test

The model learns from 316 records. It is scored on 79 other records. The baseline predicts the average final grade in the training records for everyone in the test set.

An R summary describes the training fit. For performance on unseen records, use `reports/r/metrics.csv`, which is calculated from predictions on the test rows.

## 5. Read the error metrics

| Metric | Meaning | Better direction |
|---|---|---|
| MAE | Average absolute prediction error | Lower |
| RMSE | Square root of the average squared error; responds more strongly to large misses | Lower |
| R² | 1 minus squared prediction error divided by squared deviations from the test-target mean | Higher |

If actual grades are `[10, 12]` and predictions are `[9, 14]`, absolute errors are `[1, 2]`, so MAE is 1.5 grade points. RMSE is `sqrt((1² + 2²) / 2)`, about 1.58.

R² can be negative. Our baseline uses the training mean, while the R² denominator uses the test mean. A small difference between those means explains the slightly negative baseline R² here.

## 6. Connect Python and R

| Step | Python | R |
|---|---|---|
| Load a semicolon CSV | `pd.read_csv(path, sep=";")` | `read.csv(path, sep=";")` |
| Simple regression | `LinearRegression().fit(train[["G2"]], train["G3"])` | `lm(G3 ~ G2, data = train)` |
| Multiple regression | `LinearRegression().fit(train[["G1", "G2"]], train["G3"])` | `lm(G3 ~ G1 + G2, data = train)` |
| Make predictions | `model.predict(test[features])` | `predict(model, newdata = test)` |

The notebook also solves the same problem with `np.linalg.lstsq`. An intercept column of ones is included in the design matrix. This connects regression directly to linear algebra without explicitly inverting a matrix.

## 7. Questions to practise answering

1. Why must G3 be excluded from input features?
2. Why do the two languages read the same split file?
3. Why is it misleading to call R² an accuracy percentage?
4. What do the large negative residuals in the test plot mean?
5. Why must G2 already be available when making a prediction?
6. Would these results tell us how the model performs at IUST? What additional evidence would be needed?

Try the example command, inspect a few rows in the saved predictions, and explain one large error in terms of actual versus predicted values. Avoid changing the model based on individual test errors and then reporting the same test score as an independent assessment.
