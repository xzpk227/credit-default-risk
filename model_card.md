# Model Card: Credit Default Risk Model

## Model Details

| Field | Value |
|---|---|
| Model type | XGBoost (gradient boosted trees) |
| Task | Binary classification — predict 90+ day delinquency within 2 years |
| Training data | Give Me Some Credit (Kaggle, 2011), ~150,000 borrowers |
| Features | 10 original + 6 engineered (delinquency severity, utilization segment, etc.) |
| Output | Probability of default [0, 1]; mapped to risk band (Low / Medium / High) |
| Evaluation metric | ROC-AUC (primary), Average Precision, Brier Score (calibration) |

---

## Intended Use

- **Primary use case**: Retail credit application scoring for unsecured lending (credit cards, personal loans).
- **Intended users**: Credit risk analysts, model risk validators, underwriting teams.
- **Deployment context**: Pre-screening of loan applications; not intended as the sole decision-maker.

---

## Performance

| Metric | Logistic Regression | Scorecard (WoE+LR) | XGBoost | LightGBM |
|---|---|---|---|---|
| ROC-AUC | ~0.85 | ~0.86 | ~0.87 | ~0.87 |
| Avg Precision | ~0.45 | ~0.46 | ~0.49 | ~0.49 |
| Brier Score | ~0.08 | ~0.08 | ~0.07 | ~0.07 |

*Approximate values on 20% holdout set; see Notebook 2 for exact figures.*

---

## Risk Bands & Approval Policy

| Risk Band | Default Probability | Recommendation | Rationale |
|---|---|---|---|
| Low | < 10% | Approve | Below typical lender risk appetite threshold |
| Medium | 10% – 25% | Manual Review | Marginal; human judgment or additional verification warranted |
| High | ≥ 25% | Decline | Exceeds acceptable risk; adverse action notice required |

Thresholds should be calibrated to the lender's risk appetite, regulatory requirements, and target approval rate. These are illustrative defaults only.

---

## Limitations

1. **Data vintage**: Training data is from 2011. Borrower behavior and macroeconomic conditions have changed materially.
2. **US-centric**: The dataset reflects US consumer credit patterns; model may underperform in other geographies.
3. **Self-reported income**: `MonthlyIncome` is self-reported and not verified. Income imputation for ~19% of records introduces uncertainty.
4. **Class imbalance**: ~6.7% default rate; model uses `scale_pos_weight` to compensate but may still underestimate defaults in extreme tail.
5. **No time-series features**: The model does not capture account age, spending trends, or payment velocity — features typically available in production systems.
6. **Fairness**: This model card does not include a fairness analysis. Age and income are included as features; lenders must ensure compliance with the Equal Credit Opportunity Act (ECOA) and fair lending regulations. Disparate impact analysis is required before production deployment.

---

## Ethical Considerations

- Model outputs should be one input to a decision, not the sole determinant.
- Adverse action reasons must be provided to declined applicants (FCRA / Regulation B requirements).
- Regular performance monitoring and model re-validation are required (typically annually or after material portfolio shifts).
- Explainability via SHAP is included to support model risk validation and regulatory review (SR 11-7 guidance).

---

## Caveats for Model Risk Validation (SR 11-7)

- This model is a proof-of-concept built for portfolio learning and GitHub demonstration.
- It has **not** been through a formal model validation process.
- Production deployment would require: independent validation, backtesting, benchmarking against challenger models, stress testing, and ongoing performance monitoring.
