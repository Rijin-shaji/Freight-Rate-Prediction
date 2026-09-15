# Freight-Rate-Prediction

Machine learning project for predicting trucking freight rates using historical load and market data.

## Overview

This project predicts `posted_rate` for freight loads using features such as:

* Pickup and delivery locations
* Distance
* Equipment type
* Weight
* Date and calendar features
* Market signals

The dataset contains **48,000 historical loads** from January–October 2025, with a separate **12,000-load validation set** covering November–December 2025.

## Approach

* Performed data cleaning and median-based missing-value imputation.
* Created a combined **lane** feature from pickup and delivery locations.
* Added calendar features such as month, day of week, weekend, and holiday proximity.
* Used ordinal encoding for categorical features.
* Applied `log1p` transformation to the target because `posted_rate` is right-skewed.
* Trained an **XGBoost regression model** to capture the non-linear relationship between distance and freight rates.
* Used a separate feature set/model for the December forecast because market signals and coordinates were unavailable in the December input data.

## Model Performance

| Metric |       Score |
| ------ | ----------: |
| MAE    | **$111.70** |
| RMSE   | **$533.76** |
| MAPE   |   **5.23%** |
| R²     |   **0.867** |

Distance was the most important feature, accounting for **77.1%** of the model's feature importance.

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib**

## Key Insight

Freight rate is not simply proportional to distance. Shorter lanes tend to have a higher rate per mile due to fixed operating costs, making a non-linear model more suitable for this problem.

## Future Improvements

* Time-based validation
* Better lane representation using embeddings
* Prediction intervals
* More historical data for learning seasonal and holiday effects

* ## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
'''
