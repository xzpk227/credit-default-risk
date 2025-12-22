import pandas as pd
import numpy as np


FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
TARGET_COL = "SeriousDlqin2yrs"

# Caps derived from domain knowledge + EDA (99th percentile).
OUTLIER_CAPS = {
    "RevolvingUtilizationOfUnsecuredLines": 1.0,
    "age": 100,
    "NumberOfTime30-59DaysPastDueNotWorse": 13,
    "DebtRatio": 5.0,
    "MonthlyIncome": 50_000,
    "NumberOfOpenCreditLinesAndLoans": 40,
    "NumberOfTimes90DaysLate": 13,
    "NumberRealEstateLoansOrLines": 10,
    "NumberOfTime60-89DaysPastDueNotWorse": 13,
    "NumberOfDependents": 10,
}


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df[[TARGET_COL] + FEATURE_COLS].copy()
    return df


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = missing / len(df) * 100
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct}).query(
        "missing_count > 0"
    )


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # MonthlyIncome: impute with median (skewed distribution)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    # NumberOfDependents: impute with 0 (most borrowers report 0)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, cap in OUTLIER_CAPS.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)
    # age lower bound: drop borrowers under 18
    df = df[df["age"] >= 18].reset_index(drop=True)
    return df


def clean(path: str) -> pd.DataFrame:
    df = load_raw(path)
    df = impute_missing(df)
    df = cap_outliers(df)
    return df
