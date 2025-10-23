# Detailed Model Proposals for Airline Rating Prediction

## 1. Introduction

This document outlines a detailed strategy for selecting, developing, and evaluating machine learning models to predict airline `Rating`. The prediction task is a multi-class classification problem, as evidenced by the use of `stratify=y` during data splitting in `src/data/process.py`.

The feature engineering in `src/data/pre_process.py` has produced a rich, fully numerical dataset suitable for a wide range of models. The feature set includes:
- **Ordinal Features**: `Class_Encoded`.
- **One-Hot Encoded (Sparse) Features**: `Traveller_Type_*`, `Verified_*`, `Start_Cont_*`, `End_Cont_*`.
- **Frequency Encoded Features**: `Passanger_Name`, `Start_Country_freq`, `End_Country_freq`, `Start_Code_freq`, `End_Code_freq`.
- **Numerical Features**: `Sentiment_Compound` (from review text), `Has_Layover`.

This variety of feature types informs the selection of appropriate models and pre-processing steps like feature scaling.

## 2. Primary Model Candidates

Here are detailed proposals for models that are well-suited for this task.

### 2.1. Gradient Boosting Machines (LightGBM / XGBoost)

- **Model Overview**: These are highly efficient, state-of-the-art ensemble models that build sequential decision trees, where each new tree corrects the errors of the previous one. They are the dominant approach for winning competitions on tabular data.
- **Relevance to Project**:
    - **Performance**: Unmatched performance on structured/tabular data.
    - **Feature Handling**: They internally handle mixed data types and do not require feature scaling, which simplifies the pipeline. They can effectively use the ordinal, frequency-encoded, and one-hot encoded features as-is.
    - **Complexity**: They can capture very complex, non-linear relationships and high-order feature interactions, which are likely present in this dataset (e.g., interaction between `Class` and `Sentiment_Compound`).
- **Key Hyperparameters to Tune**:
    - `n_estimators`: Number of boosting rounds/trees.
    - `learning_rate`: Step size shrinkage to prevent overfitting.
    - `max_depth`: Maximum depth of individual trees.
    - `num_leaves` (LGBM only): Controls the complexity of the trees.
    - `reg_alpha` (L1) & `reg_lambda` (L2): Regularization terms.

### 2.2. Random Forest

- **Model Overview**: An ensemble model that fits multiple decision trees on different sub-samples of the dataset and uses averaging to improve predictive accuracy and control over-fitting.
- **Relevance to Project**:
    - **Robustness**: Less prone to overfitting than a single decision tree and generally requires less hyperparameter tuning than Gradient Boosting models.
    - **Excellent Baseline**: It's a powerful model that serves as a very strong baseline. Its performance can often be close to that of a tuned GBM.
    - **Feature Handling**: Like GBMs, it does not require feature scaling and works well with the existing feature set.
- **Key Hyperparameters to Tune**:
    - `n_estimators`: Number of trees in the forest.
    - `max_depth`: Maximum depth of the trees.
    - `min_samples_split`: Minimum number of samples required to split an internal node.
    - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
    - `max_features`: Number of features to consider when looking for the best split.

### 2.3. Logistic Regression

- **Model Overview**: A linear model that is simple, fast, and highly interpretable. It serves as an essential baseline to understand the data's linear separability.
- **Relevance to Project**:
    - **Interpretability**: Provides clear insights into the influence of each feature on the prediction through its coefficients.
    - **Baseline**: Establishes a performance benchmark. If complex models do not significantly outperform it, it suggests the underlying relationships are primarily linear.
    - **Critical Pre-processing**: **Requires feature scaling** (e.g., `StandardScaler`) to ensure all features are on a similar scale. This is crucial for regularization to work correctly and for the model to converge.
- **Key Hyperparameters to Tune**:
    - `C`: Inverse of regularization strength. Smaller values specify stronger regularization.
    - `penalty`: The type of regularization (`l1`, `l2`, `elasticnet`).

### 2.4. Multi-Layer Perceptron (MLP)

- **Model Overview**: A simple feedforward neural network. It can learn complex, non-linear decision boundaries.
- **Relevance to Project**:
    - **Capturing Non-linearity**: Can potentially uncover intricate patterns that tree-based models might miss.
    - **Data Readiness**: The all-numerical feature set is ready for use, but **feature scaling is mandatory** for neural networks to perform well. `StandardScaler` or `MinMaxScaler` should be applied.
    - **Architecture**: A simple starting architecture could be 2-3 hidden layers with `ReLU` activation functions and a `softmax` output layer for multi-class classification.
- **Key Hyperparameters to Tune**:
    - `hidden_layer_sizes`: The number of neurons and layers (e.g., `(64, 32)`).
    - `activation`: Activation function for hidden layers (`relu`, `tanh`).
    - `solver`: The solver for weight optimization (`adam` is a good default).
    - `alpha`: L2 penalty (regularization) term.
    - `learning_rate_init`: Initial learning rate.

## 3. Secondary Model Candidates

### 3.1. CatBoost

- **Model Overview**: A modern gradient boosting variant that excels with categorical features.
- **Relevance to Project**: While all features have been manually encoded, CatBoost could be tested on a version of the data with minimal encoding (i.e., leaving categorical columns as is). It often provides superior results by using its own sophisticated internal encoding methods for categorical variables.

### 3.2. Support Vector Machines (SVM)

- **Model Overview**: A powerful classifier that finds an optimal hyperplane to separate classes. With non-linear kernels (like `rbf`), it can model complex relationships.
- **Relevance to Project**:
    - **Effective in High Dimensions**: Can perform well even with a large number of features.
    - **Pre-processing**: Like Logistic Regression and MLPs, **feature scaling is essential**.
    - **Considerations**: Can be computationally expensive to train on very large datasets.
- **Key Hyperparameters to Tune**:
    - `C`: Regularization parameter.
    - `kernel`: The kernel to use (`linear`, `poly`, `rbf`).
    - `gamma`: Kernel coefficient for `rbf` and `poly`.

## 4. Evaluation Strategy

To ensure a robust and fair comparison of the models, the following evaluation strategy is proposed:

1.  **Cross-Validation**: Use **k-fold cross-validation** (e.g., k=5 or k=10) on the training set (`X_train`, `y_train`) for both hyperparameter tuning and model evaluation. This provides a reliable estimate of model performance and reduces the risk of overfitting.
2.  **Evaluation Metrics**: Since this is a multi-class classification problem and class imbalance may be a factor, a suite of metrics should be used:
    - **Accuracy**: For a general sense of performance.
    - **Weighted F1-Score**: Accounts for class imbalance and provides a balanced measure of precision and recall.
    - **Classification Report**: Per-class precision, recall, and F1-score to understand performance on each `Rating` category.
    - **Confusion Matrix**: To visualize which classes are being confused with each other.

## 5. Proposed Experimental Workflow

1.  **Establish Baseline**: Train a `LogisticRegression` and a `RandomForest` with default settings and evaluate using 5-fold cross-validation. This sets the performance floor.
2.  **Tune Primary Candidates**: Use `RandomizedSearchCV` with a wide range of hyperparameters for `LightGBM` and `MLP` to find promising parameter regions.
3.  **Refine Tuning**: Follow up with `GridSearchCV` on a narrower range of parameters for the best-performing model(s) from the previous step.
4.  **Final Model Selection**: Select the model with the best cross-validated F1-score.
5.  **Test Set Evaluation**: Train the final, tuned model on the *entire* training set (`X_train`, `y_train`) and perform a single, final evaluation on the hold-out test set (`X_test`, `y_test`). **This result should be reported as the final model performance.**