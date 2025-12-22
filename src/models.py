from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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
