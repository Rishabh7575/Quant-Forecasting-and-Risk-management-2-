# Preprocessing Summary Report

This report summarizes the results and decisions made during the preprocessing stage of the Financial Transaction Risk & Anomaly Engine.

## Preprocessing Statistics

| Metric | Value | Detail |
|---|---|---|
| **Original Shape** | (10000, 8) | Shape of raw data loaded from `data/raw/transactions.csv` |
| **Final Shape** | (10000, 8) | Shape of cleaned data saved to `data/processed/transactions_clean.csv` |
| **Duplicates Removed** | 0 | Checked duplicate rows across all features; none found |
| **Missing Values Handled** | 0 | Checked all columns for missing entries; none found |
| **Invalid Values Removed** | 0 | Checked transaction amount <= 0; all amounts > 0 |

## Preprocessing Decisions and Actions

### 1. Data Type Validation & Conversion
* **Timestamp**: Converted from string (`object`) to `datetime64[ns]` format. This enables downstream temporal feature extraction (hour, day of week) without parsing latency.
* **Amount**: Enforced as a numeric float.
* **Target (`is_anomaly`)**: Enforced as integer.
* **Categorical / Identifiers**: Enforced as strings (`object`) to prevent accidental type coercion.

### 2. Categorical Standardization
* **Features**: `merchant_category`, `location`, `device_type`
* **Action**: Handled leading/trailing whitespaces (stripped) and converted values to lowercase. Standardizing string inputs prevents duplicate category representation (e.g. "Web" vs "web", " online_retail" vs "online_retail") and guarantees consistent one-hot encoding or label representation.

### 3. Duplicate and Missing Value Strategy (Implemented but not triggered)
* **Duplicate Rows**: Duplicates are removed using `drop_duplicates()`.
* **Missing Values**:
  * Numeric columns: Filled with column median.
  * Categorical columns: Filled with column mode.
  * Key columns and Target column: Rows dropped if missing.
  Since the synthetic raw dataset is fully complete, no rows were altered during this run.

### 4. Invalid Value Detection
* Transaction amounts must be strictly positive. Any transactions with `amount <= 0` are automatically filtered out. All 10,000 transactions were verified to have positive amounts (minimum value: $1.00).

---
**Report generated on:** 2026-07-27
