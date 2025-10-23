### ~~~ GLOBAL IMPORTS ~~~ ###
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, TypeAlias

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

### ~~~ LOCAL IMPORTS ~~~ ###

### ~~~ CUSTOM TYPES ~~~ ###
tensor_t: TypeAlias = np.ndarray


@dataclass
class EvaluationResult:
    model_name: str
    test_accuracy: float
    test_weighted_f1: float
    cross_val_accuracy: float
    cross_val_weighted_f1: float
    classification_summary: str
    confusion: tensor_t


model_runner_t: TypeAlias = Callable[
    [tensor_t, tensor_t, tensor_t, tensor_t, list[str]], EvaluationResult
]

### ~~~ STATE DEFINITIONS ~~~ ###
DATA_PATH = "dbs/cooked/data.npz"
MODEL_CHECKPOINT_PATH = "src/models/best_model.keras"
REPORTS_PATH = "reports/explainability"
L2_REG = 0.01
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
EPOCHS = 1080
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
CV_FOLDS = 5
AVAILABLE_MODELS = (
    "mlp",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svm",
)
SHOW_PLOTS = True

### ~~~ FUNCTION DEFINITIONS ~~~ ###


def load_data(path: str) -> Tuple[tensor_t, tensor_t, tensor_t, tensor_t, list[str]]:
    """
    Load training and testing data from a NumPy binary file.

    Args:
        path: The path to the NumPy `.npz` file containing the dataset.

    Returns:
        A tuple containing the training features, training labels, testing
        features, testing labels, and feature names.
    """
    with np.load(path, allow_pickle=True) as data:
        x_train = data["X_train"]
        y_train = data["y_train"]
        x_test = data["X_test"]
        y_test = data["y_test"]
        feature_names = data["X_col_names"].tolist()
    return x_train, y_train, x_test, y_test, feature_names


def build_mlp_model(
    input_shape: Tuple[int, ...], l2_reg: float, dropout_rate: float
) -> Model:
    """
    Build the baseline multi-layer perceptron classifier.

    Args:
        input_shape: The shape of the input feature tensor.
        l2_reg: The L2 regularization factor to apply to dense layers.
        dropout_rate: The dropout probability for regularization.

    Returns:
        An uncompiled Keras model implementing the baseline architecture.
    """
    inputs = Input(shape=input_shape)
    x = Dense(128, activation="relu", kernel_regularizer=l2(l2_reg))(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model: Model = Model(inputs=inputs, outputs=outputs)
    return model


def compute_cross_validation_metrics(
    estimator: Pipeline, x_train: tensor_t, y_train: tensor_t
) -> Tuple[float, float]:
    """
    Compute mean cross-validation metrics for a scikit-learn estimator.

    Args:
        estimator: The estimator or pipeline to evaluate.
        x_train: The training feature tensor.
        y_train: The training label tensor.

    Returns:
        A tuple containing the mean accuracy and weighted F1-score across
        the configured cross-validation folds.
    """
    scores = cross_validate(
        estimator,
        x_train,
        y_train,
        cv=CV_FOLDS,
        scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
        n_jobs=1,
    )
    cv_accuracy = float(np.mean(scores["test_accuracy"]))
    cv_weighted_f1 = float(np.mean(scores["test_f1_weighted"]))
    return cv_accuracy, cv_weighted_f1


def evaluate_predictions(
    model_name: str,
    y_true: tensor_t,
    y_pred: tensor_t,
    cross_val_accuracy: float,
    cross_val_weighted_f1: float,
) -> EvaluationResult:
    """
    Build an evaluation summary from ground truth and predicted labels.

    Args:
        model_name: A human-readable identifier for the evaluated model.
        y_true: The ground-truth target labels.
        y_pred: The predicted labels.
        cross_val_accuracy: Mean cross-validation accuracy for the model.
        cross_val_weighted_f1: Mean cross-validation weighted F1-score.

    Returns:
        A structured evaluation result containing cumulative metrics.
    """
    test_accuracy = float(accuracy_score(y_true, y_pred))
    test_weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    summary = classification_report(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    return EvaluationResult(
        model_name=model_name,
        test_accuracy=test_accuracy,
        test_weighted_f1=test_weighted_f1,
        cross_val_accuracy=cross_val_accuracy,
        cross_val_weighted_f1=cross_val_weighted_f1,
        classification_summary=summary,
        confusion=matrix,
    )


def train_evaluate_sklearn_pipeline(
    model_name: str,
    pipeline: Pipeline,
    x_train: tensor_t,
    y_train: tensor_t,
    x_test: tensor_t,
    y_test: tensor_t,
) -> EvaluationResult:
    """
    Train and evaluate a scikit-learn pipeline following the project protocol.

    Args:
        model_name: A descriptive identifier for the pipeline.
        pipeline: The pipeline to train and evaluate.
        x_train: Training features.
        y_train: Training labels.
        x_test: Testing features.
        y_test: Testing labels.

    Returns:
        An evaluation summary capturing cross-validation and test metrics.
    """
    cross_val_accuracy, cross_val_weighted_f1 = compute_cross_validation_metrics(
        pipeline, x_train, y_train
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    return evaluate_predictions(
        model_name,
        y_test,
        y_pred,
        cross_val_accuracy,
        cross_val_weighted_f1,
    )


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Construct the logistic regression pipeline with feature scaling.

    Returns:
        A scikit-learn pipeline combining standard scaling and logistic regression.
    """
    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    return pipeline


def build_random_forest_pipeline() -> Pipeline:
    """
    Construct the random forest classification pipeline.

    Returns:
        A scikit-learn pipeline wrapping the configured random forest classifier.
    """
    classifier = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("classifier", classifier)])
    return pipeline


def build_gradient_boosting_pipeline() -> Pipeline:
    """
    Construct the gradient boosting classification pipeline.

    Returns:
        A scikit-learn pipeline wrapping the gradient boosting classifier.
    """
    classifier = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=300,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    pipeline = Pipeline(steps=[("classifier", classifier)])
    return pipeline


def build_svm_pipeline() -> Pipeline:
    """
    Construct the support vector machine classification pipeline.

    Returns:
        A scikit-learn pipeline combining feature scaling with an SVM classifier.
    """
    classifier = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    return pipeline


def run_shap_analysis(
    model: Model, x_train: tensor_t, x_test: tensor_t, feature_names: list[str]
) -> None:
    """
    Run SHAP analysis on the MLP model and save a summary plot.

    Args:
        model: The trained Keras model.
        x_train: The training feature tensor.
        x_test: The testing feature tensor.
        feature_names: The names of the features.
    """
    print("--- Running SHAP Analysis ---")
    number_of_samples = 50
    # Use a subset of the training data for the explainer background
    background = x_train[
        np.random.choice(x_train.shape[0], number_of_samples, replace=False)
    ]
    explainer = shap.KernelExplainer(
        lambda data: model.predict(data, verbose=0), background
    )

    # Use a subset of the test data for SHAP value calculation
    x_test_subset = x_test[
        np.random.choice(x_test.shape[0], number_of_samples, replace=False)
    ]
    shap_values = explainer.shap_values(x_test_subset)

    if shap_values.ndim == 3:
        shap_values = np.squeeze(shap_values, axis=-1)

    # --- Global Feature Importance ---
    plt.figure(figsize=(10, 10))
    shap.summary_plot(
        shap_values,
        x_test_subset,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()

    if SHOW_PLOTS:
        plt.show()
    else:
        plot_path = os.path.join(REPORTS_PATH, "shap_summary.png")
        plt.savefig(plot_path)
        print(f"SHAP summary plot saved to {plot_path}")
    plt.close()

    # --- Local Feature Importance ---
    i = 0
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[i, :],
        x_test_subset[i, :],
        feature_names=feature_names,
    )

    if SHOW_PLOTS:
        # The default force_plot is interactive and needs to be saved to HTML
        plot_path = os.path.join(REPORTS_PATH, "shap_force_plot.html")
        shap.save_html(plot_path, force_plot)
        print(f"SHAP force plot saved to {plot_path}. Open this file in a browser to view.")
    else:
        plot_path = os.path.join(REPORTS_PATH, "shap_force_plot.html")
        shap.save_html(plot_path, force_plot)
        print(f"SHAP force plot saved to {plot_path}")


def run_lime_analysis(
    model: Model,
    x_train: tensor_t,
    x_test: tensor_t,
    y_train: tensor_t,
    feature_names: list[str],
) -> None:
    """
    Run LIME analysis on the MLP model and save an explanation.

    Args:
        model: The trained Keras model.
        x_train: The training feature tensor.
        x_test: The testing feature tensor.
        y_train: The training label tensor.
        feature_names: The names of the features.
    """
    print("--- Running LIME Analysis ---")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        x_train,
        feature_names=feature_names,
        class_names=["class_0", "class_1"],
        discretize_continuous=True,
    )

    def predict_fn_for_lime(x: np.ndarray) -> np.ndarray:
        """Wrapper for model.predict to format output for LIME."""
        predictions = model.predict(x, verbose=0)
        return np.hstack([1 - predictions, predictions])

    # Explain a single instance from the test set
    i = np.random.randint(0, x_test.shape[0])
    exp = explainer.explain_instance(
        x_test[i],
        predict_fn_for_lime,
        num_features=len(feature_names),
    )

    if SHOW_PLOTS:
        plt.figure()
        exp.as_pyplot_figure()
        plt.tight_layout()
        plt.show()
    else:
        report_path = os.path.join(REPORTS_PATH, "lime_report.html")
        exp.save_to_file(report_path)
        print(f"LIME report saved to {report_path}")


def run_mlp_model(
    x_train: tensor_t,
    y_train: tensor_t,
    x_test: tensor_t,
    y_test: tensor_t,
    feature_names: list[str],
) -> EvaluationResult:
    """
    Train and evaluate the baseline multi-layer perceptron classifier.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_test: Testing features.
        y_test: Testing labels.
        feature_names: The names of the features.

    Returns:
        An evaluation result describing the MLP performance.
    """
    model = build_mlp_model(x_train.shape[1:], L2_REG, DROPOUT_RATE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(
        MODEL_CHECKPOINT_PATH,
        save_best_only=True,
        monitor="val_loss",
    )
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0,
    )
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)  # type: ignore
    predictions = model.predict(x_test, verbose=0)
    y_pred = (predictions.flatten() >= 0.5).astype(int)

    # --- Explainability Analysis ---
    run_shap_analysis(model, x_train, x_test, feature_names)
    run_lime_analysis(model, x_train, x_test, y_train, feature_names)

    result = evaluate_predictions("MLP", y_test, y_pred, float(np.nan), float(np.nan))
    result.test_accuracy = float(accuracy)
    result.test_weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    return result


def run_logistic_regression_model(
    x_train: tensor_t,
    y_train: tensor_t,
    x_test: tensor_t,
    y_test: tensor_t,
    feature_names: list[str],
) -> EvaluationResult:
    """
    Train and evaluate the logistic regression model.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_test: Testing features.
        y_test: Testing labels.
        feature_names: The names of the features.

    Returns:
        The evaluation summary for logistic regression.
    """
    pipeline = build_logistic_regression_pipeline()
    return train_evaluate_sklearn_pipeline(
        "Logistic Regression", pipeline, x_train, y_train, x_test, y_test
    )


def run_random_forest_model(
    x_train: tensor_t,
    y_train: tensor_t,
    x_test: tensor_t,
    y_test: tensor_t,
    feature_names: list[str],
) -> EvaluationResult:
    """
    Train and evaluate the random forest model.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_test: Testing features.
        y_test: Testing labels.
        feature_names: The names of the features.

    Returns:
        The evaluation summary for random forest classification.
    """
    pipeline = build_random_forest_pipeline()
    return train_evaluate_sklearn_pipeline(
        "Random Forest", pipeline, x_train, y_train, x_test, y_test
    )


def run_gradient_boosting_model(
    x_train: tensor_t,
    y_train: tensor_t,
    x_test: tensor_t,
    y_test: tensor_t,
    feature_names: list[str],
) -> EvaluationResult:
    """
    Train and evaluate the gradient boosting model.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_test: Testing features.
        y_test: Testing labels.
        feature_names: The names of the features.

    Returns:
        The evaluation summary for gradient boosting classification.
    """
    pipeline = build_gradient_boosting_pipeline()
    return train_evaluate_sklearn_pipeline(
        "Gradient Boosting", pipeline, x_train, y_train, x_test, y_test
    )


def run_svm_model(
    x_train: tensor_t,
    y_train: tensor_t,
    x_test: tensor_t,
    y_test: tensor_t,
    feature_names: list[str],
) -> EvaluationResult:
    """
    Train and evaluate the support vector machine model.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_test: Testing features.
        y_test: Testing labels.
        feature_names: The names of the features.

    Returns:
        The evaluation summary for SVM classification.
    """
    pipeline = build_svm_pipeline()
    return train_evaluate_sklearn_pipeline(
        "Support Vector Machine", pipeline, x_train, y_train, x_test, y_test
    )


def get_model_registry() -> Dict[str, model_runner_t]:
    """
    Build the registry mapping model identifiers to execution functions.

    Returns:
        A dictionary linking model keys to their respective runner functions.
    """
    return {
        "mlp": run_mlp_model,
        "logistic_regression": run_logistic_regression_model,
        "random_forest": run_random_forest_model,
        "gradient_boosting": run_gradient_boosting_model,
        "svm": run_svm_model,
    }


def display_evaluation_result(result: EvaluationResult) -> None:
    """
    Pretty-print the evaluation metrics for a trained model.

    Args:
        result: The evaluation summary to display.
    """
    cross_val_accuracy = (
        "N/A"
        if np.isnan(result.cross_val_accuracy)
        else f"{result.cross_val_accuracy:.4f}"
    )
    cross_val_weighted_f1 = (
        "N/A"
        if np.isnan(result.cross_val_weighted_f1)
        else f"{result.cross_val_weighted_f1:.4f}"
    )
    print("=" * 80)
    print(f"Model: {result.model_name}")
    print(f"Test Accuracy: {result.test_accuracy:.4f}")
    print(f"Test Weighted F1: {result.test_weighted_f1:.4f}")
    print(f"CV Accuracy: {cross_val_accuracy}")
    print(f"CV Weighted F1: {cross_val_weighted_f1}")
    print("Classification Report:")
    print(result.classification_summary)
    print("Confusion Matrix:")
    print(result.confusion)


def main() -> None:
    """
    Main script execution function selecting, training, and evaluating models.
    """
    # --- Create reports directory ---
    os.makedirs(REPORTS_PATH, exist_ok=True)

    x_train, y_train, x_test, y_test, feature_names = load_data(DATA_PATH)
    model_registry = get_model_registry()
    for model_key in get_model_registry().keys():
        runner = model_registry[model_key]
        result = runner(x_train, y_train, x_test, y_test, feature_names)
        display_evaluation_result(result)


### ~~~ SCRIPT EXECUTION ~~~ ###
if __name__ == "__main__":
    main()
