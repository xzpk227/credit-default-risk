import optuna
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)


def logistic_regression_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            random_state=42,
        )),
    ])


def scorecard_logistic_regression() -> LogisticRegression:
    # No scaling: WoE-encoded features are already on a comparable scale
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=0.05,
        random_state=42,
    )


def xgboost_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=6,   # ~6:1 class imbalance in Give Me Some Credit
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def tune_xgboost(X, y, n_trials: int = 50, cv: int = 5) -> XGBClassifier:
    """Use Optuna to find the best XGBoost hyperparameters via cross-validation."""
    from sklearn.metrics import roc_auc_score
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    X_arr = np.array(X)
    y_arr = np.array(y)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "scale_pos_weight":  scale_pos_weight,
            "eval_metric":       "auc",
            "random_state":      42,
            "n_jobs":            -1,
            "verbosity":         0,
        }
        aucs = []
        for train_idx, val_idx in skf.split(X_arr, y_arr):
            model = XGBClassifier(**params)
            model.fit(X_arr[train_idx], y_arr[train_idx])
            prob = model.predict_proba(X_arr[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_arr[val_idx], prob))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest AUC (CV): {study.best_value:.4f}")
    print(f"Best params:   {study.best_params}")

    best_model = XGBClassifier(
        **study.best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    return best_model, study


class PlattCalibratedModel:
    """
    Platt scaling: fit a logistic regression on top of a trained model's
    predicted probabilities using a held-out calibration set.
    """
    def __init__(self, base_model):
        self.base_model = base_model
        self.calibrator = LogisticRegression()

    def fit(self, X_calib, y_calib):
        raw_prob = self.base_model.predict_proba(X_calib)[:, 1].reshape(-1, 1)
        self.calibrator.fit(raw_prob, y_calib)
        return self

    def predict_proba(self, X):
        raw_prob = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal_prob = self.calibrator.predict_proba(raw_prob)
        return cal_prob


def calibrate_model(model, X_calib, y_calib) -> PlattCalibratedModel:
    """
    Apply Platt scaling on top of a trained model.
    Uses a held-out calibration set — NOT the training set — to avoid overfitting.
    """
    calibrated = PlattCalibratedModel(model)
    calibrated.fit(X_calib, y_calib)
    return calibrated


def lightgbm_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
