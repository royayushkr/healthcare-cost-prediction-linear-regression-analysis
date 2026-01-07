# Healthcare Cost Prediction Linear Regression Analysis

## Project Overview
This project aims to predict individual medical costs billed by health insurance. By analyzing a dataset of insurance customers, the model identifies patterns in healthcare expenses related to personal attributes like age, BMI, and smoking habits. The goal is to provide a predictive tool for validation on unseen customer data.


## Dataset
The analysis uses `insurance.csv`, containing demographic and lifestyle details of insurance beneficiaries.

**Key Features:**
* **Demographics:** `age`, `sex`, `region`.
* **Health Indicators:** `bmi` (Body Mass Index), `smoker` (smoking status).
* **Dependents:** `children` (number of dependents).
* **Target Variable:** `charges` (individual medical costs).

## Methodology

### 1. Data Cleaning
* **Text Normalization:** Standardized `sex` values (e.g., converting "man" to "male", "woman" to "female") and `region` names to lowercase.
* **Data Correction:** Removed non-numeric characters (like `$`) from the `charges` column and converted it to float.
* **Outlier Handling:** Filtered out invalid entries such as negative age or negative children counts.

### 2. Model Pipeline
The project implements a machine learning pipeline using `scikit-learn`:
* **Feature Engineering:** Converted categorical variables (`sex`, `smoker`, `region`) into dummy variables using one-hot encoding.
* **Scaling:** Applied `StandardScaler` to normalize numerical features (`age`, `bmi`, `children`).
* **Modeling:** Trained a `LinearRegression` model on the processed dataset.

### 3. Model Evaluation & Validation
* **Performance:** Evaluated the model using 5-fold cross-validation, achieving a Mean R² score of approximately **0.75**.
* **Validation:** Applied the trained model to a separate `validation_dataset.csv` to generate cost predictions for new customers.
* **Post-Processing:** Enforced a minimum predicted charge of **$1000** to ensure realistic output values.

## Results
The linear regression model successfully predicts healthcare charges with a strong correlation to the input features.
* **Mean R² Score:** ~0.745
* **Validation Output:** Generated a dataframe of predicted charges for the validation set, with values adjusted to a minimum threshold.

## Technologies Used
* **Python**: Core programming language.
* **pandas**: Data manipulation and cleaning.
* **scikit-learn**: Machine learning pipeline (`Pipeline`, `StandardScaler`, `LinearRegression`, `cross_val_score`).
* **numpy**: Numerical operations.

## Usage
1.  Ensure `insurance.csv` and `validation_dataset.csv` are in the project directory.
2.  Run the Jupyter Notebook to execute the cleaning and training pipeline.
3.  The notebook will output model performance metrics and display the validation data with new `predicted_charges`.
