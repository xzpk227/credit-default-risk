import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Delinquency severity score (mirrors the SQL query)
    df["delinquency_severity"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] * 1
        + df["NumberOfTime60-89DaysPastDueNotWorse"] * 2
        + df["NumberOfTimes90DaysLate"] * 3
    )

    # Total past-due events
    df["total_past_due_events"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )

    # Any severe delinquency flag
    df["has_90day_late"] = (df["NumberOfTimes90DaysLate"] > 0).astype(int)

    # Utilization segment (mirrors SQL)
    df["utilization_segment"] = pd.cut(
        df["RevolvingUtilizationOfUnsecuredLines"],
        bins=[-np.inf, 0.2, 0.5, 0.9, np.inf],
        labels=["low", "moderate", "high", "maxed_out"],
    )

    # Age bucket
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 24, 34, 49, 64, np.inf],
        labels=["18-24", "25-34", "35-49", "50-64", "65+"],
    )

    # Debt burden: monthly debt payment estimate
    df["monthly_debt_est"] = df["DebtRatio"] * df["MonthlyIncome"]

    # Credit line density: open lines per real-estate loan (proxy for credit mix)
    df["credit_line_density"] = df["NumberOfOpenCreditLinesAndLoans"] / (
        df["NumberRealEstateLoansOrLines"] + 1
    )

    return df


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    exclude = {"SeriousDlqin2yrs", "utilization_segment", "age_bucket", "borrower_id"}
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


# ---------------------------------------------------------------------------
# Weight of Evidence (WoE) encoding for scorecard-style model
# ---------------------------------------------------------------------------

class WoEEncoder:
    """Bin numeric features and compute Weight of Evidence for each bin."""

    def __init__(self, n_bins: int = 10, min_samples: int = 50):
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.woe_maps: dict[str, pd.Series] = {}
        self.iv_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        total_events = y.sum()
        total_non_events = (1 - y).sum()

        for col in X.columns:
            bins = pd.qcut(X[col], q=self.n_bins, duplicates="drop")
            grouped = pd.DataFrame({"bin": bins, "y": y}).groupby("bin")["y"]
            events = grouped.sum()
            non_events = grouped.count() - events

            # Clip to avoid log(0)
            event_rate = (events / total_events).clip(lower=1e-6)
            non_event_rate = (non_events / total_non_events).clip(lower=1e-6)

            woe = np.log(event_rate / non_event_rate)
            iv = ((event_rate - non_event_rate) * woe).sum()

            self.woe_maps[col] = woe
            self.iv_[col] = iv

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        for col in X.columns:
            if col not in self.woe_maps:
                result[col] = X[col]
                continue
            bins = pd.qcut(X[col], q=self.n_bins, duplicates="drop", retbins=False)
            result[f"{col}_woe"] = bins.map(self.woe_maps[col]).astype(float).fillna(0)
        return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def iv_summary(self) -> pd.DataFrame:
        return (
            pd.DataFrame.from_dict(self.iv_, orient="index", columns=["IV"])
            .sort_values("IV", ascending=False)
        )
