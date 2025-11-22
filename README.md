# Gold Recovery Optimization: Predictive Modeling

## Project Description

This Data Science project focuses on developing a **predictive machine learning model** to optimize the industrial gold recovery process (flotation), which is a critical factor in mining profitability. The goal is to forecast the amount of gold recovered in the key stages of the process, enabling the company to **make informed operational decisions** to maximize efficiency and precious metal yield.

The model's performance is rigorously evaluated using the industry-standard metric, the **sMAPE (Symmetric Mean Absolute Percentage Error)**, weighted to reflect the significance of different processing phases.

***

## Project Goal

To develop a Machine Learning model capable of predicting **Gold Recovery** at two critical stages of the flotation process:
1.  **Rougher Stage (Primary Flotation)**
2.  **Final Stage (Final Concentrate)**

***

## Key Evaluation Metric (Total sMAPE)

The model's quality was assessed using a custom metric, the **sMAPE (Symmetric Mean Absolute Percentage Error)**, calculated for each stage and then combined with a weight defined by the mining company:

$$\text{Total sMAPE} = 25\% \times \text{sMAPE}_{\text{Rougher}} + 75\% \times \text{sMAPE}_{\text{Final}}$$

The high 75% weighting on the **Final** stage underscores the importance of prediction accuracy for the final product yield.

***

## Methodology and Data Analysis

### 1. Data Preprocessing and Verification

* **Data Integrity:** The accuracy of the provided rougher recovery column was verified against the official recovery formula. The Mean Absolute Error (MAE) between the calculated and actual recovery was **0.0**, confirming data integrity.
* **Data Alignment:** Missing target values (recovery) in the test set (`df_test`) were successfully retrieved and merged from the full dataset (`df_full`) using the timestamp (`date`) as the key.
* **Outlier Handling:** Records where metal concentration or recovery was **close to zero** were filtered out. These values were assumed to represent measurement errors or plant shutdowns to prevent bias during model training.
* **Imputation:** Missing values were handled using the **forward fill** method, assuming that input status parameters remain constant between measurements. This preserves the time-series nature of the data, which is crucial for model accuracy.

### 2. Modeling and Evaluation

* **Technique:** A **Cross-Validation (CV)** workflow was implemented to evaluate model performance on the weighted Total sMAPE metric, training separate models for the *rougher* and *final* stage predictions.
* **Baseline Models:** A **Linear Regression** model was compared against a basic **Random Forest Regressor**.
    * Linear Regression (Total sMAPE CV): **7.29%**
    * Random Forest Regressor (Total sMAPE CV): **6.75%**
* **Winning Model:** The **Random Forest Regressor** demonstrated the best performance in Cross-Validation and was selected as the final model.

***

## Key Results

The final model (Random Forest Regressor) was tested against the dedicated test set, yielding the following performance on the weighted Total sMAPE metric:

| Concentration Stage | sMAPE (Test Set) | Weighting |
| :--- | :--- | :--- |
| **Rougher** (Primary) | 10.27% | 25% |
| **Final** (Final Product) | 12.57% | 75% |
| **Weighted Total sMAPE** | **11.99%** | 100% |

The higher error observed in the **Final** stage suggests greater complexity and uncertainty in the variables influencing the final product yield.

***

## Professional Recommendations for Future Work

To further enhance predictive accuracy and reduce the overall sMAPE, the following are strongly recommended:

* **Advanced Algorithms:** Explore **Gradient Boosting** algorithms (such as XGBoost, LightGBM). These models often outperform Random Forest in regression problems and could potentially mitigate the higher error observed in the final stage.
* **Feature Engineering:** Calculate and test new features, such as the **total concentration** of all metals (Au, Ag, Pb, Sol) at each input and output stage. These cumulative metrics may prove highly predictive.
* **Hyperparameter Tuning:** Conduct a more extensive search (e.g., Grid Search or Randomized Search) to optimize the parameters of the chosen Random Forest Regressor or the selected Gradient Boosting model.

***

## Technologies Used

* **Language:** Python
* **Key Libraries:**
    * `pandas`, `numpy`
    * `matplotlib.pyplot`
    * `sklearn.metrics` (`mean_absolute_error`)
    * `sklearn.model_selection` (`cross_val_score`, `KFold`)
    * `sklearn.linear_model` (`LinearRegression`)
    * `sklearn.ensemble` (`RandomForestRegressor`)
    * `sklearn.preprocessing` (`StandardScaler`)