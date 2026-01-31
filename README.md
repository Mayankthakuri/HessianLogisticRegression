# HessianLogisticRegression
Hessian Logistic Regression for Breast Cancer Prediction

## Project Overview
This project implements a custom `HessianLogisticRegression` model from scratch, utilizing Newton's method for optimization, and applies it to the Breast Cancer Wisconsin diagnostic dataset. The goal is to predict whether a tumor is benign or malignant based on various features, demonstrating the model's capabilities in a real-world binary classification scenario.

## Model Implementation: HessianLogisticRegression
The `HessianLogisticRegression` class is a custom implementation of logistic regression using Newton's method with L2 regularization. Key methods include:
-   `__init__(self, lambda_reg=0.1, max_iter=100, tol=1e-6)`: Initializes model parameters, including the regularization strength (`lambda_reg`), maximum iterations, and convergence tolerance.
-   `_add_bias(self, X)`: A helper method to prepend a column of ones to the feature matrix `X`, accommodating the intercept term.
-   `_sigmoid(self, z)`: Implements the sigmoid (logistic) function, robustly clipping input values for numerical stability.
-   `predict_proba(self, X)`: Calculates the predicted probabilities of the positive class for given input features.
-   `predict(self, X)`: Returns binary class labels (0 or 1) based on the predicted probabilities and a 0.5 threshold.
-   `fit(self, X, y)`: Trains the model using Newton's method. It iteratively computes gradients and Hessians (with L2 regularization) to update the weights (`beta`) until convergence or maximum iterations are reached. It also records convergence information (negative log-likelihood, delta_beta norm).
-   `get_convergence_info(self)`: Returns a list of dictionaries containing training history, including iteration number, negative log-likelihood, and the norm of the weight update (`delta_beta_norm`).

## Dataset: Breast Cancer Wisconsin
The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset, obtained from KaggleHub (`uciml/breast-cancer-wisconsin-data`). It contains features computed from digitized images of fine needle aspirates (FNAs) of breast masses. The target variable is 'diagnosis', indicating whether the tumor is Malignant (M) or Benign (B).

## Data Preprocessing
The following steps were performed to prepare the dataset for modeling:
1.  **Loading**: The `data.csv` file was loaded into a pandas DataFrame.
2.  **Column Dropping**: Unnecessary columns such as 'id' and 'Unnamed: 32' were removed.
3.  **Target Mapping**: The 'diagnosis' column was mapped from categorical values ('M', 'B') to numerical (1 for Malignant, 0 for Benign).
4.  **Missing Values**: Checked for and confirmed no missing values after dropping 'Unnamed: 32'.
5.  **Splitting**: The dataset was split into features (X) and target (y), and then into training and testing sets using `train_test_split` (80% training, 20% testing).
6.  **Feature Scaling**: Numerical features were scaled using `StandardScaler` to ensure all features contribute equally to the model's training process.

## Exploratory Data Analysis (EDA)
Key findings from the EDA include:
-   **Missing Values**: No missing values were found in the processed DataFrame.
-   **Class Distribution**: The 'diagnosis' column showed a class imbalance: 357 (62.74%) benign cases (0) and 212 (37.26%) malignant cases (1).
-   **Feature Distributions**: Histograms of selected features (`radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`) provided insights into their distribution characteristics.
-   **Correlations**: Analysis of the correlation matrix revealed strong positive correlations between the 'diagnosis' target variable and features like `concave points_worst` (0.7936), `perimeter_worst` (0.7829), `concave points_mean` (0.7766), `radius_worst` (0.7765), `perimeter_mean` (0.7426), `area_worst` (0.7338), `radius_mean` (0.7300), and `area_mean` (0.7090).

## Hyperparameter Tuning
A grid search approach with 5-fold cross-validation (`KFold`) was employed to fine-tune the `lambda_reg` hyperparameter for the `HessianLogisticRegression` model. The following `lambda_reg` values were evaluated: `[0.001, 0.01, 0.1, 1.0, 10.0]`.

The tuning process identified `lambda_reg = 1.0` (and `10.0`) as the optimal value, yielding the best average cross-validation accuracy of **0.9780**.

## Model Training and Evaluation
(This section will be updated with the final model training details, including the selected `lambda_reg`, and comprehensive evaluation metrics such as accuracy, precision, recall, F1-score, and the confusion matrix on the test set.)
"""
