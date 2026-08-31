# Daily Retail Revenue Prediction using Machine Learning

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting daily retail revenue using historical transaction data.

The project uses the UCI Online Retail Dataset, which contains transaction-level information such as invoice number, product, quantity, unit price, invoice date, customer, and country.

The transaction-level data is cleaned and aggregated into daily revenue. A continuous daily time series is then created, followed by time-series feature engineering using calendar features, lag features, and rolling averages.

A Random Forest Regression model is trained to predict daily retail revenue. Since the data is time-dependent, chronological train-validation-test splitting and TimeSeriesSplit cross-validation are used instead of random splitting.

---

## Objectives

The main objectives of this project are:

- Analyze and inspect the retail transaction dataset.
- Clean and preprocess the raw transaction data.
- Convert transaction-level data into daily revenue.
- Create a continuous daily time series.
- Perform time-series feature engineering.
- Split the dataset chronologically into training, validation, and test sets.
- Train a baseline Random Forest Regression model.
- Analyze feature importance.
- Perform hyperparameter tuning using time-series cross-validation.
- Train a final tuned Random Forest model.
- Compare baseline and tuned model performance.
- Perform prediction error analysis.
- Visualize actual and predicted revenue.

---

## Dataset

### Dataset Used

UCI Online Retail Dataset

The original dataset contains:

- 541,909 transaction records
- 8 attributes
- 25,900 unique invoices
- 4,372 unique customers

### Dataset Attributes

| Attribute | Description |
|---|---|
| InvoiceNo | Invoice number |
| StockCode | Product/item code |
| Description | Product description |
| Quantity | Quantity purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit |
| CustomerID | Customer identifier |
| Country | Customer country |

### Date Range

The dataset contains transactions from:

2010-12-01 to 2011-12-09

---

## Project Workflow

The complete machine learning workflow is:

UCI Online Retail Dataset
          |
          v
    Data Inspection
          |
          v
     Data Cleaning
          |
          v
 Daily Revenue Aggregation
          |
          v
Continuous Daily Time Series
          |
          v
  Feature Engineering
          |
          v
Chronological Data Splitting
          |
          v
Baseline Random Forest
          |
          v
Feature Importance Analysis
          |
          v
Hyperparameter Tuning
          |
          v
Final Tuned Random Forest
          |
          v
Model Comparison
          |
          v
Error Analysis
          |
          v
     Visualization

Data Inspection

The original dataset was inspected to determine:

Dataset dimensions
Column names
Data types
Missing values
Date range
Number of unique invoices
Number of unique customers

The original dataset shape was:

541909 rows × 8 columns

Missing values were mainly observed in:

Description
CustomerID

The dataset contained:

25,900 unique invoices
4,372 unique customers
Data Cleaning

The following preprocessing operations were performed:

The Online Retail Excel dataset was loaded.
Cancelled transactions were identified.
Cancelled transactions were removed.
Records with invalid quantity or price values were removed.
Transaction revenue was calculated using:
Revenue = Quantity × UnitPrice
Revenue was aggregated by date.
The resulting daily retail revenue dataset was saved as a CSV file.
Cleaning Results
Original records:               541909
Cancelled transactions:           9288
Invalid quantity/price records:   2517

The processed daily dataset contained:

305 daily observations

with a date range from:

2010-12-01 to 2011-12-09
Continuous Time-Series Preparation

The aggregated daily dataset did not contain an entry for every calendar day.

Therefore, a continuous daily time series was created by generating all calendar dates between the minimum and maximum dates.

Days for which no sales transaction was recorded were assigned:

Daily_Revenue = 0
Results
Original daily observations:     305
Continuous daily observations:   374
Zero-sales days:                  69
Missing revenue values:            0

The resulting continuous dataset was saved as:

data/continuous_daily_sales.csv
Feature Engineering

Time-series and calendar-based features were created to improve the model's ability to predict daily revenue.

The final feature-engineered dataset contains 360 observations.

Features Used
Feature	Description
Day_of_Week	Day of the week
Day_of_Month	Calendar day of the month
Month	Month number
Week_of_Year	Week number of the year
Lag_1	Revenue from the previous day
Lag_7	Revenue from 7 days earlier
Lag_14	Revenue from 14 days earlier
Rolling_7_Day_Mean	Mean revenue over the previous 7 days
Target Variable
Daily_Revenue

The final feature-engineered dataset was saved as:

data/final_feature_engineered_sales.csv
Temporal Train / Validation / Test Split

Because retail revenue is time-dependent, a random train-test split was not used.

The final feature-engineered dataset was divided chronologically into training, validation, and test sets.

Training Set
Observations: 251
Start: 2010-12-15
End:   2011-08-22
Validation Set
Observations: 55
Start: 2011-08-23
End:   2011-10-16
Test Set
Observations: 54
Start: 2011-10-17
End:   2011-12-09

The chronological ordering was verified successfully:

Training < Validation < Test

The final datasets were saved as:

data/ml_train.csv
data/ml_validation.csv
data/ml_test.csv
Time-Series Cross-Validation

Five folds of TimeSeriesSplit were used for model evaluation and hyperparameter tuning.

Unlike ordinary random cross-validation, TimeSeriesSplit preserves chronological order.

The model is trained using earlier observations and validated using later observations.

The training window progressively expands across the folds.

This prevents future observations from being used to train a model that is evaluated on earlier observations.

Baseline Random Forest Model

A baseline Random Forest Regression model was trained using the engineered features.

Features Used
Day_of_Week
Day_of_Month
Month
Week_of_Year
Lag_1
Lag_7
Lag_14
Rolling_7_Day_Mean
Baseline Validation Performance
Metric	Value
MAE	11,832.75
RMSE	17,825.08
R²	0.3743
Baseline Test Performance
Metric	Value
MAE	20,713.04
RMSE	31,040.85
R²	0.0926

The baseline predictions were saved as:

data/random_forest_predictions.csv
Feature Importance Analysis

Feature importance was calculated using the trained Random Forest model.

Feature Importance Results
Feature	Importance
Day_of_Week	0.304722
Lag_7	0.303739
Rolling_7_Day_Mean	0.099354
Lag_1	0.080090
Week_of_Year	0.071684
Day_of_Month	0.063851
Lag_14	0.057262
Month	0.019298

The most important features were:

Day_of_Week
Lag_7
Rolling_7_Day_Mean
Lag_1

The results indicate that weekly patterns and recent historical revenue have a strong influence on daily revenue predictions.

The feature importance visualization was saved as:

data/feature_importance.png
Hyperparameter Tuning

Hyperparameter tuning was performed using TimeSeriesSplit cross-validation.

The best hyperparameters obtained were:

n_estimators = 100
max_depth = 10
min_samples_split = 2
min_samples_leaf = 4
Best Cross-Validation Performance
Best Cross-Validation MAE = 7913.95

The tuned model showed improved validation performance compared with the baseline model.

Final Tuned Random Forest Model

The final Random Forest model was trained using the best hyperparameters obtained during tuning.

Validation Performance
Metric	Value
MAE	11,595.92
RMSE	17,508.30
R²	0.3964
Test Performance
Metric	Value
MAE	18,918.01
RMSE	29,559.63
R²	0.1771

The final predictions were saved as:

data/final_model_predictions.csv
Baseline vs Tuned Model
Model	Validation MAE	Validation RMSE	Validation R²	Test MAE	Test RMSE	Test R²
Baseline Random Forest	11832.75	17825.08	0.3743	20713.04	31040.85	0.0926
Tuned Random Forest	11595.92	17508.30	0.3964	18918.01	29559.63	0.1771
Performance Improvement

The tuned Random Forest improved upon the baseline model.

Test MAE decreased from:

20,713.04 → 18,918.01

Test RMSE decreased from:

31,040.85 → 29,559.63

Test R² increased from:

0.0926 → 0.1771

Therefore, the tuned model achieved better performance than the baseline model on the final test dataset.

The comparison results were saved as:

data/model_comparison.csv
Error Analysis

Error analysis was performed using the final tuned model predictions.

Final Test Error Metrics
MAE  = 18,918.01
RMSE = 29,559.63
Largest Prediction Errors

The largest prediction error occurred on:

Date:              2011-12-09
Actual Revenue:    200,920.60
Predicted Revenue: 47,297.10
Absolute Error:    153,623.50

Other large errors occurred on days with unusually high revenue values.

The model generally captures recurring weekly revenue patterns but has difficulty predicting sudden and unusually large revenue spikes.

The error analysis results were saved as:

data/final_error_analysis.csv
Visualizations

The project generates several visualizations.

1. Actual vs Predicted Revenue

This visualization compares actual daily revenue with the revenue predicted by the Random Forest model.

File:

data/final_model_actual_vs_predicted.png
2. Feature Importance

This visualization displays the relative importance of each feature used by the Random Forest model.

File:

data/feature_importance.png
3. Baseline vs Tuned Model Performance

This visualization compares the R² scores of the baseline and tuned Random Forest models on validation and test datasets.

File:

data/model_comparison_r2.png
Project Structure
ML_Self_Learning/
│
├── data/
│   ├── Online Retail.xlsx
│   ├── daily_retail_sales.csv
│   ├── continuous_daily_sales.csv
│   ├── feature_engineered_sales.csv
│   ├── final_feature_engineered_sales.csv
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   ├── ml_train.csv
│   ├── ml_validation.csv
│   ├── ml_test.csv
│   ├── random_forest_predictions.csv
│   ├── tuned_validation_predictions.csv
│   ├── final_model_predictions.csv
│   ├── model_comparison.csv
│   ├── final_error_analysis.csv
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   ├── final_model_actual_vs_predicted.png
│   └── model_comparison_r2.png
│
├── inspect_data.py
├── clean_data.py
├── prepare_timeseries.py
├── feature_engineering.py
├── split_data.py
├── split_features.py
├── train_model.py
├── feature_importance.py
├── tune_model.py
├── final_model.py
├── model_comparison.py
├── error_analysis.py
├── visualize_results.py
│
└── README.md
Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Scikit-learn
Matplotlib
OpenPyXL
Machine Learning Techniques
Random Forest Regression
Time-Series Cross-Validation
Hyperparameter Tuning
Feature Importance Analysis
Regression Evaluation
Error Analysis
Evaluation Metrics

The following metrics were used to evaluate the regression model.

Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted revenue.

MAE = Average(|Actual - Predicted|)

Lower MAE indicates better performance.

Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

RMSE = √(Average((Actual - Predicted)²))

RMSE gives greater importance to large prediction errors.

R² Score

R² measures how much of the variation in the target variable is explained by the model.

A higher R² generally indicates better predictive performance.

How to Run the Project
1. Create a Virtual Environment
python -m venv venv
2. Activate the Virtual Environment

For Windows:

venv\Scripts\activate
3. Install Required Libraries
pip install pandas numpy scikit-learn matplotlib openpyxl
4. Place the Dataset

Place the downloaded dataset:

Online Retail.xlsx

inside:

data/
5. Run the Scripts

Run the scripts in the following order:

python inspect_data.py
python clean_data.py
python prepare_timeseries.py
python feature_engineering.py
python split_features.py
python train_model.py
python feature_importance.py
python tune_model.py
python final_model.py
python model_comparison.py
python error_analysis.py
python visualize_results.py
Results

The final tuned Random Forest model achieved the following test performance:

MAE  : 18,918.01
RMSE : 29,559.63
R²   : 0.1771

Compared with the baseline model:

Baseline Test MAE : 20,713.04
Tuned Test MAE    : 18,918.01
Baseline Test RMSE : 31,040.85
Tuned Test RMSE    : 29,559.63
Baseline Test R² : 0.0926
Tuned Test R²    : 0.1771

The tuned model therefore provides an improvement over the baseline Random Forest model.

Conclusion

This project demonstrates an end-to-end machine learning workflow for daily retail revenue prediction using historical transaction data.

The implementation covers:

Dataset inspection
Data cleaning
Transaction-level revenue calculation
Daily revenue aggregation
Continuous time-series preparation
Time-series feature engineering
Chronological train-validation-test splitting
Time-series cross-validation
Random Forest regression
Feature importance analysis
Hyperparameter tuning
Final model training
Model comparison
Error analysis
Data visualization

The tuned Random Forest model improved the test R² from 0.0926 to 0.1771 and reduced the test MAE from 20,713.04 to 18,918.01 compared with the baseline model.

The results also show that the model is able to capture recurring patterns in retail revenue, particularly weekly behavior and recent historical revenue. However, unusually large revenue spikes remain difficult to predict accurately.

Future Improvements

The project can be further improved by:

Testing additional regression algorithms such as Gradient Boosting, XGBoost, or LightGBM.
Adding additional lag features.
Adding holiday and special-event indicators.
Adding more seasonal and trend-based features.
Applying transformations to highly skewed revenue values.
Experimenting with advanced time-series forecasting models.
Performing more detailed residual analysis.
Incorporating additional business-related features that may explain sudden revenue spikes.
