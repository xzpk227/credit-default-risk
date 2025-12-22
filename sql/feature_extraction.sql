-- Credit Default Risk: Feature Extraction Queries
-- These queries run against the SQLite database at data/processed/credit.db.
-- In notebook 01 they are executed via pandas.read_sql_query().
-- Column names use backtick quoting because some contain hyphens.

-- ============================================================
-- 1. Base borrower profile
-- ============================================================
SELECT
    borrower_id,
    age,
    MonthlyIncome,
    NumberOfDependents,
    RevolvingUtilizationOfUnsecuredLines,
    DebtRatio,
    NumberOfOpenCreditLinesAndLoans,
    NumberRealEstateLoansOrLines,
    SeriousDlqin2yrs AS target
FROM borrowers
WHERE age BETWEEN 18 AND 100;

-- ============================================================
-- 2. Delinquency severity score
-- ============================================================
SELECT
    borrower_id,
    (`NumberOfTime30-59DaysPastDueNotWorse` * 1
     + `NumberOfTime60-89DaysPastDueNotWorse` * 2
     + NumberOfTimes90DaysLate * 3) AS delinquency_severity_score,
    CASE
        WHEN NumberOfTimes90DaysLate > 0                    THEN 'severe'
        WHEN `NumberOfTime60-89DaysPastDueNotWorse` > 0     THEN 'moderate'
        WHEN `NumberOfTime30-59DaysPastDueNotWorse` > 0     THEN 'mild'
        ELSE 'none'
    END AS delinquency_category
FROM borrowers;

-- ============================================================
-- 3. Utilization and income segments
-- ============================================================
SELECT
    borrower_id,
    RevolvingUtilizationOfUnsecuredLines,
    CASE
        WHEN RevolvingUtilizationOfUnsecuredLines > 0.9 THEN 'maxed_out'
        WHEN RevolvingUtilizationOfUnsecuredLines > 0.5 THEN 'high_utilization'
        WHEN RevolvingUtilizationOfUnsecuredLines > 0.2 THEN 'moderate_utilization'
        ELSE 'low_utilization'
    END AS utilization_segment,
    CASE
        WHEN MonthlyIncome IS NULL   THEN 'unknown'
        WHEN MonthlyIncome < 2000    THEN 'low_income'
        WHEN MonthlyIncome < 5000    THEN 'mid_income'
        WHEN MonthlyIncome < 10000   THEN 'high_income'
        ELSE 'very_high_income'
    END AS income_segment
FROM borrowers;

-- ============================================================
-- 4. Age buckets
-- ============================================================
SELECT
    borrower_id,
    age,
    CASE
        WHEN age < 25              THEN '18-24'
        WHEN age BETWEEN 25 AND 34 THEN '25-34'
        WHEN age BETWEEN 35 AND 49 THEN '35-49'
        WHEN age BETWEEN 50 AND 64 THEN '50-64'
        ELSE '65+'
    END AS age_bucket
FROM borrowers;

-- ============================================================
-- 5. Default rate by age bucket (analytical query)
-- ============================================================
SELECT
    CASE
        WHEN age < 25              THEN '18-24'
        WHEN age BETWEEN 25 AND 34 THEN '25-34'
        WHEN age BETWEEN 35 AND 49 THEN '35-49'
        WHEN age BETWEEN 50 AND 64 THEN '50-64'
        ELSE '65+'
    END AS age_bucket,
    COUNT(*)                              AS total_borrowers,
    SUM(SeriousDlqin2yrs)                AS total_defaults,
    ROUND(AVG(SeriousDlqin2yrs) * 100, 2) AS default_rate_pct
FROM borrowers
GROUP BY age_bucket
ORDER BY age_bucket;

-- ============================================================
-- 6. Final feature join for model input
-- ============================================================
WITH base AS (
    SELECT
        borrower_id,
        age,
        MonthlyIncome,
        NumberOfDependents,
        RevolvingUtilizationOfUnsecuredLines,
        DebtRatio,
        NumberOfOpenCreditLinesAndLoans,
        NumberRealEstateLoansOrLines,
        `NumberOfTime30-59DaysPastDueNotWorse`,
        `NumberOfTime60-89DaysPastDueNotWorse`,
        NumberOfTimes90DaysLate,
        SeriousDlqin2yrs AS target
    FROM borrowers
    WHERE age BETWEEN 18 AND 100
),
delinquency AS (
    SELECT
        borrower_id,
        (`NumberOfTime30-59DaysPastDueNotWorse` * 1
         + `NumberOfTime60-89DaysPastDueNotWorse` * 2
         + NumberOfTimes90DaysLate * 3) AS delinquency_severity_score
    FROM borrowers
)
SELECT
    b.*,
    d.delinquency_severity_score,
    CASE
        WHEN b.RevolvingUtilizationOfUnsecuredLines > 0.9 THEN 'maxed_out'
        WHEN b.RevolvingUtilizationOfUnsecuredLines > 0.5 THEN 'high_utilization'
        WHEN b.RevolvingUtilizationOfUnsecuredLines > 0.2 THEN 'moderate_utilization'
        ELSE 'low_utilization'
    END AS utilization_segment,
    CASE
        WHEN b.age < 25              THEN '18-24'
        WHEN b.age BETWEEN 25 AND 34 THEN '25-34'
        WHEN b.age BETWEEN 35 AND 49 THEN '35-49'
        WHEN b.age BETWEEN 50 AND 64 THEN '50-64'
        ELSE '65+'
    END AS age_bucket
FROM base b
JOIN delinquency d ON b.borrower_id = d.borrower_id;
