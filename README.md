# StaySense Analytics 🏨

A comprehensive **Time Series Forecasting and Machine Learning project for analyzing and predicting daily scheduled hotel reservation arrivals** using Python, Scikit-learn, Statsmodels, and PyTorch.

The project focuses on understanding historical reservation patterns, preparing daily time series data, engineering temporal and historical features, developing and comparing forecasting models, evaluating prediction performance, and explaining forecasting behavior through ensemble analysis and feature importance.

The forecasting framework predicts scheduled reservation arrivals over a **7-day horizon**, combining statistical forecasting, machine learning, and deep learning approaches.

---

## 🌟 Features

### Core Features

- **Data Understanding & Cleaning:** Inspection of reservation records, dataset structure, missing values, duplicates, and arrival-date consistency.
- **Exploratory Data Analysis:** Investigation of reservation distributions, arrival patterns, temporal trends, and daily reservation variability.
- **Time Series Preparation:** Conversion of reservation-level records into a chronological daily scheduled-arrival series.
- **Feature Engineering:** Development of calendar, lag, rolling-window, expanding-window, and trend features.
- **Forecasting Models:** Development and comparison of statistical, machine learning, and deep learning forecasting approaches.
- **Hyperparameter Optimization:** Model configuration tuning using chronological validation.
- **Ensemble Forecasting:** Integration of LSTM, SARIMA, and Extra Trees predictions through weighted averaging.
- **Time Series Cross-Validation:** Expanding-window validation with a 7-day forecasting horizon.
- **Model Evaluation:** Assessment using MAE, RMSE, WAPE, and R².
- **Forecast Error Analysis:** Investigation of forecast errors across prediction horizons and reservation-volume levels.
- **Model Explainability:** Analysis of ensemble components, prediction agreement, and Extra Trees feature importance.
- **Business Insights:** Identification of opportunities for reservation-volume planning and forecasting-informed operational decisions.

---

## 🗂 Project Structure

The project is organized as follows:

```text
StaySense-Analytics/
│
├── data/
│   ├── raw/                                  # Original hotel reservation dataset
│   └── processed/                            # Cleaned and prepared datasets
│       ├── cleaned_reservations.csv
│       ├── daily_reservations.csv
│       ├── train_daily.csv
│       ├── test_daily.csv
│       ├── feature_engineered_daily.csv
│       ├── train_features.csv
│       └── test_features.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb           # Dataset inspection
│   ├── 02_exploratory_data_analysis.ipynb    # Exploratory Data Analysis
│   ├── 03_data_preparation.ipynb             # Cleaning and chronological splitting
│   ├── 04_feature_engineering.ipynb          # Temporal and historical features
│   ├── 05_model_development.ipynb            # Forecasting model experimentation
│   ├── 06_model_evaluation.ipynb             # Holdout evaluation and error analysis
│   └── 07_model_explainability.ipynb         # Ensemble and feature interpretation
│
├── models/                                   # Saved model and explainability artifacts
│
├── artifacts/                                # Model configurations and analysis outputs
│   ├── selected_model_configurations.json
│   ├── model_test_predictions.csv
│   ├── model_test_comparison.csv
│   ├── final_model_evaluation.csv
│   ├── forecast_error_analysis.csv
│   ├── forecast_horizon_performance.csv
│   ├── reservation_level_performance.csv
│   ├── ensemble_component_weights.csv
│   ├── ensemble_component_forecasts.csv
│   ├── ensemble_component_performance.csv
│   ├── ensemble_component_agreement.csv
│   ├── ensemble_agreement_summary.csv
│   ├── extra_trees_feature_importance.csv
│   └── extra_trees_feature_group_importance.csv
│
├── reports/
│   └── figures/                              # Project visualizations
│
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
└── .gitignore                                # Git ignored files
```

**Note:** Processed datasets, model configurations, evaluation outputs, and explainability artifacts are retained to support reproducibility. Some trained models are reconstructed from their saved configurations rather than relying exclusively on serialized model files.

---

## 🏗️ Project Architecture

```text
+--------------------------------------+
|              DATA LAYER              |
|                                      |
|     Hotel Reservation Dataset        |
|     Reservation-Level Records        |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|          DATA UNDERSTANDING          |
|                                      |
|  - Dataset inspection                |
|  - Missing values                    |
|  - Duplicate analysis                |
|  - Arrival-date validation           |
|  - Reservation characteristics       |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|              EDA LAYER               |
|                                      |
|  - Reservation distributions         |
|  - Daily arrival patterns            |
|  - Temporal trends                   |
|  - Calendar patterns                 |
|  - Reservation variability           |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|           DATA PREPARATION           |
|                                      |
|  - Invalid-date removal              |
|  - Daily aggregation                 |
|  - Chronological ordering            |
|  - Train-test splitting              |
|  - Processed artifact creation       |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|          FEATURE ENGINEERING         |
|                                      |
|  - Calendar features                 |
|  - Historical lag features           |
|  - Rolling statistics                |
|  - Expanding statistics              |
|  - Historical trend features         |
|  - Leakage-aware feature creation    |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|           MODEL DEVELOPMENT          |
|                                      |
|  - Statistical forecasting           |
|  - Machine learning                  |
|  - Deep learning                     |
|  - Expanding-window validation       |
|  - Hyperparameter optimization       |
|  - Ensemble forecasting              |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|           MODEL EVALUATION           |
|                                      |
|  - Independent holdout testing       |
|  - Rolling-origin forecasts          |
|  - MAE, RMSE, WAPE, R²               |
|  - Forecast-horizon analysis         |
|  - Reservation-volume analysis       |
|  - Prediction error diagnostics      |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|          MODEL EXPLAINABILITY        |
|                                      |
|  - Ensemble weight analysis          |
|  - Component forecast reconstruction |
|  - Individual model performance      |
|  - Component prediction agreement    |
|  - Extra Trees feature importance    |
|  - Feature group contributions       |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
|            BUSINESS INSIGHTS         |
|                                      |
|  - Reservation-volume patterns       |
|  - Short-term forecasting            |
|  - Operational planning support      |
|  - Forecast uncertainty              |
|  - Model limitations                 |
+--------------------------------------+
```

---

## 🛠 Tech Stack

- **Language:** Python 3.11
- **Environment:** Jupyter Notebook, VS Code
- **Libraries:**
  - **Data Manipulation:** Pandas, NumPy
  - **Visualization:** Matplotlib, Seaborn
  - **Machine Learning:** Scikit-learn
  - **Statistical Forecasting:** Statsmodels
  - **Deep Learning:** PyTorch
  - **Data Preprocessing:** Scikit-learn preprocessing utilities
  - **Model Evaluation:** Scikit-learn metrics
  - **Artifact Persistence:** Joblib, JSON, CSV
- **Version Control:** Git & GitHub

---

## 📊 Dataset Overview

The project uses the **Hotel Reservation Dataset**, available on Kaggle.

The original dataset contains **36,275 reservation records** and **19 variables**, describing hotel booking characteristics and reservation outcomes.

### Dataset Features

```text
Booking_ID
no_of_adults
no_of_children
no_of_weekend_nights
no_of_week_nights
type_of_meal_plan
required_car_parking_space
room_type_reserved
lead_time
arrival_year
arrival_month
arrival_date
market_segment_type
repeated_guest
no_of_previous_cancellations
no_of_previous_bookings_not_canceled
avg_price_per_room
no_of_special_requests
booking_status
```

### Target Variable

**`scheduled_reservations`**

The target represents the number of reservation records associated with each scheduled arrival date.

The objective is to forecast daily scheduled reservation arrivals for the next **7 days** using historical daily counts and information available at each forecast origin.

**Important:** The target is derived retrospectively from reservation records. It does not represent confirmed hotel occupancy, actual guest arrivals, or a historical snapshot of bookings known at a particular point in time.

### Dataset Summary

| Component | Value |
|---|---:|
| Original reservation records | 36,275 |
| Original variables | 19 |
| Invalid arrival-date records excluded | 37 |
| Cleaned reservation records | 36,238 |
| Daily time series observations | 549 |
| Daily series start date | July 1, 2017 |
| Daily series end date | December 31, 2018 |
| Training observations | 439 days |
| Testing observations | 110 days |
| Forecast horizon | 7 days |
| Engineered predictor features | 24 |
| Feature-ready observations | 515 |

### Daily Target Statistics

| Statistic | Value |
|---|---:|
| Mean daily scheduled reservations | 66.01 |
| Standard deviation | 43.56 |
| Minimum | 1 |
| Maximum | 254 |

**Dataset Source:** [Hotel Reservation Dataset — Kaggle](https://www.kaggle.com/datasets/vanthanadevi08/hotel-reservation-dataset)

---

## 🎯 Problem Statement

Hotel reservation activity varies across dates due to calendar patterns, seasonal behavior, historical booking volumes, and other factors influencing scheduled arrivals.

Understanding these variations can support short-term operational planning, while forecasting future reservation volumes can provide a quantitative basis for anticipating periods of higher or lower scheduled arrival activity.

However, reservation-volume forecasting presents several challenges:

- Daily reservation counts fluctuate substantially.
- Historical patterns may change over time.
- Calendar effects interact with recent reservation activity.
- Forecast accuracy can deteriorate across multi-day horizons.
- Different forecasting approaches may capture different aspects of the time series.

The objective of this project is to develop a **7-day time series forecasting framework** that predicts daily scheduled reservation arrivals using historical observations, evaluates multiple forecasting approaches, and provides interpretable insights into forecasting behavior.

---

## 🚀 Project Objectives

- Understand the structure and characteristics of hotel reservation data.
- Identify historical reservation-volume patterns and temporal variations.
- Construct a consistent daily scheduled-arrival time series.
- Prepare chronological training and testing datasets.
- Engineer leakage-aware temporal and historical predictors.
- Develop statistical, machine learning, and deep learning forecasting models.
- Compare model configurations using expanding-window validation.
- Optimize selected forecasting configurations.
- Develop and evaluate weighted ensemble forecasts.
- Assess forecasting performance using multiple regression metrics.
- Investigate forecast errors across prediction horizons and reservation-volume levels.
- Explain ensemble behavior and influential tree-based forecasting features.
- Identify practical limitations and opportunities for future improvements.

---

## 🔍 Exploratory Data Analysis

The EDA stage investigates reservation characteristics and the temporal structure of scheduled arrivals.

### Analysis Areas

- Reservation-level dataset characteristics
- Reservation and arrival-date distributions
- Daily scheduled reservation volumes
- Historical reservation trends
- Monthly reservation patterns
- Day-of-week variations
- Calendar-related reservation behavior
- Daily reservation variability
- High-volume and low-volume arrival dates
- Temporal patterns relevant to forecasting

### Key Observations

- The daily scheduled-arrival series contains **549 observations** across July 2017 to December 2018.
- Daily scheduled reservation volumes average approximately **66 reservations**, with substantial variability.
- The maximum observed daily volume reaches **254 reservations**, considerably exceeding the series average.
- The variation in daily reservation volumes motivates the use of historical, calendar-based, and statistical forecasting features.

These findings informed the data preparation strategy, feature engineering approach, and selection of forecasting models.

---

## ⚙️ Data Preparation & Feature Engineering

### Data Preparation

The preparation workflow included:

- Validation of arrival-date components.
- Exclusion of 37 records containing invalid arrival dates.
- Construction of a standardized arrival datetime variable.
- Aggregation of cleaned reservation records into daily scheduled-arrival counts.
- Chronological ordering of the resulting time series.
- Creation of independent training and testing periods.
- Preservation of cleaned reservation-level and daily datasets for reuse.

### Chronological Data Splitting

The daily time series was divided chronologically to preserve temporal ordering.

| Component | Date Range | Observations |
|---|---|---:|
| Training | July 1, 2017 – September 12, 2018 | 439 |
| Testing | September 13, 2018 – December 31, 2018 | 110 |
| Total | July 1, 2017 – December 31, 2018 | 549 |

Random splitting was avoided because it would disrupt the temporal structure of the forecasting problem.

### Temporal Feature Engineering

A total of **24 predictor features** were developed across six groups.

#### Calendar Features

Calendar variables capture recurring temporal characteristics.

```text
year
quarter
month
day_of_month
day_of_week
day_of_year
week_of_year
is_weekend
is_month_start
is_month_end
```

#### Historical Lag Features

Lag variables represent historical scheduled reservation volumes.

```text
reservations_lag_7
reservations_lag_14
reservations_lag_21
reservations_lag_28
```

#### Rolling Statistics

Rolling features summarize historical reservation levels and variability.

```text
rolling_mean_7
rolling_mean_14
rolling_mean_28
rolling_std_7
rolling_std_14
rolling_std_28
```

#### Expanding Statistics

Expanding features summarize the historical series over an increasing observation window.

```text
expanding_mean
expanding_std
```

#### Trend Features

Trend variables represent changes in historical reservation levels.

```text
trend_7_day
trend_14_day
```

### Forecasting Leakage Prevention

Historical predictors were constructed to respect the 7-day forecast horizon.

This ensures that forecasting features do not depend on target observations from within the future prediction window.

### Final Feature Dataset

| Component | Value |
|---|---:|
| Total predictor features | 24 |
| Feature-ready observations | 515 |
| Training feature observations | 405 |
| Testing feature observations | 110 |
| Forecast horizon | 7 days |

The engineered datasets were saved for reuse during model development, evaluation, and explainability.

---

## 🤖 Forecasting Approach

The project treats scheduled reservation forecasting as a **supervised time series prediction problem** with a 7-day forecast horizon.

Statistical, machine learning, and deep learning approaches were explored to compare their ability to capture historical reservation patterns.

### Modeling Workflow

```text
Daily Scheduled Reservation Series
                |
                v
Chronological Train-Test Split
                |
                v
Leakage-Aware Feature Engineering
                |
                v
Multiple Forecasting Models
                |
                v
Expanding-Window Cross-Validation
                |
                v
Hyperparameter Optimization
                |
                v
Forecasting Model Comparison
                |
                v
Weighted Averaging Ensemble
                |
                v
Independent Rolling-Origin Evaluation
```

### Forecasting Model Families

The project explored multiple forecasting approaches, including:

- **Statistical Forecasting:** SARIMA configurations.
- **Machine Learning:** Tree-based regression models, including Extra Trees.
- **Deep Learning:** LSTM forecasting networks.
- **Ensemble Forecasting:** Weighted averaging of forecasts from different model families.

A total of **28 model configurations and experiments** were evaluated during development.

### Cross-Validation Strategy

The project uses **22 expanding-window validation folds** with a 7-day prediction horizon.

This strategy maintains chronological ordering while repeatedly evaluating forecasts on future observations relative to each training window.

**Mean Absolute Error (MAE)** was used as the primary model-selection metric.

### Validation Model Comparison

| Model | Validation MAE |
|---|---:|
| **Weighted Averaging** | **24.8586** |
| Tuned LSTM | 24.9624 |
| Tuned SARIMA | 25.3564 |
| Tuned Extra Trees | 25.6465 |

Weighted Averaging achieved the lowest mean validation MAE among the compared configurations.

### Selected Forecasting Model

**Weighted Averaging Ensemble**

The ensemble combines forecasts from the original LSTM, SARIMA, and Extra Trees configurations.

| Component | Ensemble Weight |
|---|---:|
| LSTM | 40% |
| SARIMA | 35% |
| Extra Trees | 25% |
| **Total** | **100%** |

The ensemble prediction is calculated as:

\[
\hat{y}_{ensemble}
=
0.40\hat{y}_{LSTM}
+
0.35\hat{y}_{SARIMA}
+
0.25\hat{y}_{ExtraTrees}
\]

**Note:** The ensemble uses the original component configurations evaluated during validation, not the individually tuned configurations listed in the model comparison table.

The ensemble was selected using validation performance before examining the independent test results.

---

## 🧠 Deep Learning Experimentation

An **LSTM (Long Short-Term Memory)** neural network was evaluated as a deep learning approach for scheduled reservation forecasting.

LSTM networks are designed to learn sequential relationships from historical observations.

### LSTM Modeling

The project investigated historical input sequences and neural network configurations for direct 7-day forecasting.

### Tuned LSTM Configuration

| Parameter | Value |
|---|---|
| Lookback window | 14 days |
| Hidden units | 16 |
| Learning rate | 0.001 |
| Weight decay | 0.001 |
| Training epochs | 100 |
| Forecast horizon | 7 days |
| Framework | PyTorch |

### Deep Learning Results

| Model | Validation MAE | Test MAE |
|---|---:|---:|
| Tuned LSTM | 24.9624 | 32.0191 |
| Weighted Averaging | **24.8586** | **29.2473** |

### Key Findings

- The LSTM approach successfully generated multi-step forecasts using historical reservation sequences.
- Tuned LSTM achieved competitive validation performance but experienced a larger error on the independent test period.
- Combining LSTM with statistical and tree-based forecasts produced the lowest validation MAE among the compared configurations.
- Deep learning did not consistently outperform the statistical or machine learning alternatives on the holdout period.

---

## 📈 Model Evaluation

The validation-selected Weighted Averaging ensemble and selected tuned models were evaluated on an independent chronological test period containing **110 daily observations**.

### Evaluation Strategy

The evaluation used **16 rolling-origin forecasting windows**.

Each window generated forecasts for up to seven future days, with the final window covering the remaining five test observations.

This preserved the intended multi-step forecasting structure throughout the holdout period.

### Final Model Performance

| Model | MAE | RMSE | WAPE | R² |
|---|---:|---:|---:|---:|
| **Tuned Extra Trees** | **29.0500** | **37.2760** | **32.8924%** | **0.2056** |
| Weighted Averaging | 29.2473 | 37.6373 | 33.1158% | 0.1901 |
| Tuned SARIMA | 30.5649 | 38.6199 | 34.6077% | 0.1473 |
| Tuned LSTM | 32.0191 | 41.4961 | 36.2543% | 0.0156 |

### Selected Ensemble Performance

| Metric | Result |
|---|---:|
| Mean Absolute Error | **29.2473** |
| Root Mean Squared Error | 37.6373 |
| Weighted Absolute Percentage Error | 33.1158% |
| R² Score | 0.1901 |
| Test observations | 110 |
| Rolling-origin windows | 16 |

### Evaluation Analysis

The evaluation included:

- Comparison of forecasting models on the independent test period.
- Actual versus predicted reservation volumes.
- Prediction error distributions.
- Forecast-horizon performance analysis.
- Reservation-volume-level performance analysis.
- Assessment of forecasting consistency.
- Comparison of validation and holdout results.

### Key Evaluation Findings

- **Tuned Extra Trees** achieved the lowest test MAE of **29.05**, narrowly outperforming Weighted Averaging.
- **Weighted Averaging** remained the designated primary model because it was selected using validation results rather than holdout performance.
- The ensemble achieved a test WAPE of **33.12%**, indicating meaningful remaining forecast error.
- The third forecast step exhibited the highest MAE across the evaluated models.
- High-volume reservation days generally produced larger absolute forecasting errors.
- Low-volume days showed relatively high percentage-based errors.

These findings indicate that forecasting performance varies across reservation levels and prediction horizons.

---

## 🔎 Model Explainability

Model explainability focused on understanding the behavior of the Weighted Averaging ensemble and identifying influential predictors within its Extra Trees component.

### Explainability Methods

- Ensemble component weight analysis.
- Reconstruction of original component forecasts.
- Verification of weighted ensemble predictions.
- Individual component performance comparison.
- Component prediction correlation analysis.
- Forecast disagreement analysis.
- Extra Trees feature importance.
- Feature group importance aggregation.

### Ensemble Forecast Reconstruction

The original LSTM, SARIMA, and Extra Trees configurations were reconstructed to reproduce the ensemble forecasts.

The reconstructed predictions were compared against the saved Weighted Averaging forecasts.

| Component | Result |
|---|---:|
| Test predictions reconstructed | 110 |
| Forecasting windows | 16 |
| Maximum absolute reconstruction difference | 0.00000000 |
| Mean absolute reconstruction difference | 0.00000000 |

The reconstructed ensemble forecasts matched the saved predictions at the displayed numerical precision.

This verifies the consistency of the original component configurations and ensemble calculation.

### Individual Ensemble Component Performance

| Model | Test MAE | Test RMSE | Test WAPE |
|---|---:|---:|---:|
| **Original SARIMA** | **29.1806** | **37.3520** | **33.0403%** |
| Weighted Averaging | 29.2473 | 37.6373 | 33.1158% |
| Original Extra Trees | 29.3094 | 37.5907 | 33.1862% |
| Original LSTM | 31.6098 | 40.9718 | 35.7908% |

### Key Component Findings

- Original SARIMA achieved the lowest individual component MAE.
- Original Extra Trees delivered performance close to the selected ensemble.
- Original LSTM recorded the highest error among the ensemble components.
- Weighted Averaging did not outperform every individual component on the test period.

### Component Prediction Agreement

Prediction correlation was examined to understand similarities between component forecasts.

| Model Pair | Prediction Correlation |
|---|---:|
| LSTM and SARIMA | 0.7632 |
| LSTM and Extra Trees | 0.6201 |
| SARIMA and Extra Trees | 0.8975 |

SARIMA and Extra Trees exhibited the strongest prediction agreement, while LSTM and Extra Trees demonstrated comparatively greater forecasting diversity.

The correlation between component disagreement and ensemble absolute error was approximately **0.0819**, indicating a weak observed relationship.

### Extra Trees Feature Importance

Feature importance was calculated using the fitted original Extra Trees configuration.

#### Top 10 Predictors

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `rolling_mean_14` | 16.35% |
| 2 | `rolling_mean_28` | 14.37% |
| 3 | `rolling_mean_7` | 9.76% |
| 4 | `year` | 9.17% |
| 5 | `day_of_week` | 6.69% |
| 6 | `expanding_mean` | 6.46% |
| 7 | `reservations_lag_14` | 5.52% |
| 8 | `week_of_year` | 3.32% |
| 9 | `reservations_lag_7` | 3.21% |
| 10 | `day_of_year` | 3.14% |

The three rolling-average features collectively accounted for approximately **40.48%** of total feature importance.

### Feature Group Importance

| Feature Group | Features | Total Importance |
|---|---:|---:|
| **Rolling Averages** | 3 | **40.48%** |
| Calendar Features | 10 | 30.50% |
| Lag Features | 4 | 12.93% |
| Expanding Statistics | 2 | 8.62% |
| Rolling Variability | 3 | 4.54% |
| Trend Features | 2 | 2.94% |

### Explainability Findings

- Rolling averages represent the most influential predictor group in the Extra Trees model.
- Calendar variables contribute substantial predictive information.
- Historical lag features provide additional information about previous reservation volumes.
- Rolling variability and trend features contribute relatively smaller shares of total importance.
- Ensemble component analysis reveals meaningful differences in forecasting behavior.

**Interpretation Note:** Extra Trees feature importance measures predictive importance within the fitted tree-based model. It does not establish causal relationships or directly explain the complete Weighted Averaging ensemble. Correlated predictors may also distribute importance across related features.

---

## 💡 Key Findings

- The hotel reservation dataset was successfully transformed into a daily scheduled-arrival forecasting series containing **549 observations**.
- Historical reservation volumes exhibit substantial daily variability, with counts ranging from **1 to 254**.
- Calendar, lag, rolling, expanding, and trend features provided a structured representation of historical reservation behavior.
- Expanding-window validation enabled chronological comparison of forecasting approaches.
- **Weighted Averaging** achieved the lowest mean validation MAE of **24.8586**.
- **Tuned Extra Trees** achieved the lowest test MAE of **29.0500**.
- The selected ensemble achieved a test MAE of **29.2473** and WAPE of **33.1158%**.
- Forecasting errors varied across prediction horizons and reservation-volume levels.
- Original SARIMA achieved the strongest individual ensemble-component performance on the test period.
- Rolling-average features accounted for **40.48%** of Extra Trees feature importance.
- The explainability analysis demonstrated that historical reservation levels and calendar patterns are influential within the tree-based forecasting component.
- The project illustrates both the potential and the limitations of short-horizon forecasting from retrospectively aggregated hotel reservation records.

---

## ⚠️ Limitations & Future Improvements

### Current Limitations

- The dataset covers a limited historical period from July 2017 to December 2018.
- The daily forecasting series contains only 549 observations, restricting the amount of historical information available for modeling.
- Scheduled reservation counts do not directly represent actual hotel occupancy or realized guest arrivals.
- The dataset does not provide historical booking snapshots needed to establish which reservations were known at each forecasting date.
- Retrospective scheduled-arrival counts therefore do not constitute a validated real-time operational forecasting dataset.
- The forecasting models do not explicitly account for all external demand drivers, such as local events, holidays, weather, or competitor activity.
- The selected ensemble achieved an R² of 0.1901 on the test period, indicating substantial unexplained variation.
- Forecast errors remain relatively high during certain reservation-volume conditions.
- Ensemble weights do not directly represent individual component importance or causal contribution.
- Tree-based feature importance can be affected by correlations among historical predictors.
- Generalization to different hotels, markets, and future time periods remains unverified.

### Future Improvements

- Extend the dataset to include longer historical periods.
- Incorporate actual booking timestamps and historical booking snapshots.
- Distinguish scheduled arrivals, cancellations, realized arrivals, and occupied rooms.
- Introduce external calendar and event information where available.
- Investigate additional seasonal and trend modeling approaches.
- Explore alternative ensemble weighting strategies using validation data.
- Evaluate forecast uncertainty through prediction intervals or probabilistic forecasting.
- Investigate methods for improving high-volume reservation forecasts.
- Assess model stability through additional chronological backtesting.
- Explore permutation importance and SHAP-based explanations where appropriate.
- Develop a production-oriented forecasting pipeline with monitoring and periodic retraining.
- Validate forecasting performance on newer reservation data and additional hotel properties.

---

## 📁 Notebook Workflow

The project is organized into seven notebooks.

| Notebook | Purpose |
|---|---|
| `01_data_understanding.ipynb` | Dataset inspection, structure, and initial quality assessment |
| `02_exploratory_data_analysis.ipynb` | Reservation patterns and temporal exploratory analysis |
| `03_data_preparation.ipynb` | Cleaning, daily aggregation, and chronological splitting |
| `04_feature_engineering.ipynb` | Calendar, lag, rolling, expanding, and trend feature creation |
| `05_model_development.ipynb` | Forecasting model development, tuning, validation, and ensemble selection |
| `06_model_evaluation.ipynb` | Independent holdout evaluation and forecasting error diagnostics |
| `07_model_explainability.ipynb` | Ensemble reconstruction, component agreement, and feature importance |

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/TejasMJ/StaySense-Analytics.git
cd StaySense-Analytics
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

#### Windows

```bash
.\.venv\Scripts\activate
```

#### Mac/Linux

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the `notebooks/` directory and execute the notebooks in numerical order.

### 6. Prepare the Dataset

Download the original Hotel Reservation Dataset from Kaggle and place the CSV file in the `data/raw/` directory using the filename expected by the data-understanding notebook.

### 7. Execute the Forecasting Workflow

Follow the seven notebooks sequentially:

1. Data Understanding
2. Exploratory Data Analysis
3. Data Preparation
4. Feature Engineering
5. Model Development
6. Model Evaluation
7. Model Explainability

**Note:** Model development includes repeated chronological validation and neural network training, so execution times may vary depending on available computing resources.

The evaluation and explainability notebooks depend on processed datasets and model artifacts generated during the preceding stages.

---

## 📌 Project Status

**Status:** Technical Implementation Completed ✅

The project includes:

- Data understanding and cleaning.
- Exploratory Data Analysis.
- Daily time series preparation.
- Chronological train-test splitting.
- Temporal and historical feature engineering.
- Statistical forecasting experimentation.
- Machine learning model development.
- LSTM deep learning experimentation.
- Expanding-window cross-validation.
- Hyperparameter optimization.
- Weighted ensemble forecasting.
- Independent rolling-origin evaluation.
- Forecast error diagnostics.
- Ensemble reconstruction and verification.
- Component prediction agreement analysis.
- Extra Trees feature importance analysis.
- Feature group interpretation.

**Final Validation-Selected Model:** Weighted Averaging Ensemble.

**Validation MAE:** 24.8586.

**Test MAE:** 29.2473.

**Test RMSE:** 37.6373.

**Test WAPE:** 33.1158%.

**Test R²:** 0.1901.

**Lowest Test MAE:** Tuned Extra Trees — 29.0500.

---

## 👨‍💻 Author

**Tejas Jadhav**

- GitHub: [@tejas-jadhav](https://github.com/TejasMJ)
- LinkedIn: [Tejas Jadhav](https://www.linkedin.com/in/tejas-m-jadhav/)
