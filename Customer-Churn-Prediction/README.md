# Financial Transaction Risk & Anomaly Engine

## Project Overview
This repository contains a production-style Machine Learning pipeline for a **Financial Transaction Risk & Anomaly Engine**. The primary objective is to monitor incoming financial transactions, run diagnostics, and detect anomalous patterns or fraudulent transactions (e.g., extremely high amounts, transaction attempts at odd hours, or international locations inconsistent with typical user history).

## Folder Structure
```text
Customer-Churn-Prediction/     # Root repository directory (repurposed for Risk & Anomaly Engine)
│
├── api/                       # API backend for model serving (FastAPI)
│   ├── main.py
│   └── requirements.txt
│
├── data/
│   ├── raw/                   # Raw immutable datasets (contains transactions.csv)
│   └── processed/             # Cleaned datasets ready for training (contains transactions_clean.csv)
│
├── logs/                      # Generated pipeline application logs
│   └── pipeline.log
│
├── src/                       # Pipeline source code
│   ├── config.py              # Central configurations and constants
│   ├── generate_synthetic_data.py  # Script to generate synthetic transaction data
│   ├── data_preprocessing.py  # Loading, validating, and cleaning pipeline
│   ├── utils.py               # Logger, loaders, and profiling summaries
│   ├── train_model.py         # Model training script (placeholder)
│   └── evaluate_model.py      # Model evaluation script (placeholder)
│
├── notebooks/                 # Prototype notebooks
│   └── churn_analysis.ipynb
│
├── reports/                   # System reports and diagnostic metrics
│   └── figures/
│
├── requirements.txt           # Main python dependencies
│   └── main.py                # Pipeline entry point orchestrator
```

## Dataset Description
The engine operates on a financial transaction dataset (`transactions.csv`), featuring:
* **Number of Rows:** 10,000 transactions.
* **Columns / Features:**
  * `transaction_id` (object): Unique alphanumeric identifier for each transaction.
  * `customer_id` (object): Alphanumeric identifier of the transacting customer.
  * `timestamp` (object): Date and time of the transaction (format: `%Y-%m-%d %H:%M:%S`).
  * `amount` (float): Transaction value.
  * `merchant_category` (object): Merchant business type (`online_retail`, `grocery`, `dining`, `travel`, `cash_withdrawal`, `transfer`).
  * `location` (object): City and country of the transaction.
  * `device_type` (object): Medium used (`mobile`, `web`, `pos`, `atm`).
  * `is_anomaly` (int): Target variable (1 for anomaly/high risk, 0 for normal transactions).

### Anomaly Signature Rules (1.5% baseline)
Anomalies in this dataset are simulated using distinct threat/risk profiles:
1. **High Value Spikes:** Large value transactions ($5,000 – $20,000) mapped to transfers or online retail.
2. **Geographical Anomalies:** Transactions originating from international destinations (e.g. London, Paris, Tokyo) for domestically-based customer cards.
3. **Odd Hours Behavior:** Large value card-present or transfer transactions initiated between 2:00 AM and 4:00 AM local time.

## Tech Stack
* **Programming Language:** Python 3.10+
* **Data Manipulation & Processing:** pandas, numpy
* **Visualization:** matplotlib, seaborn
* **Machine Learning Environment:** scikit-learn
* **APIs & Dashboards:** FastAPI, Next.js

## Getting Started

### 1. Prerequisites
Install dependencies from the root directory:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
If the raw dataset is not present, generate it using the synthetic generator:
```bash
python src/generate_synthetic_data.py
```
This writes a new `transactions.csv` to `data/raw/`.

### 3. Run Preprocessing Pipeline
Execute the main entry point to validate data constraints, run diagnostics, and save a cleaned copy to `data/processed/`:
```bash
python main.py
```
Logs are printed to stdout and saved directly to `logs/pipeline.log`.

## Planned Feature Engineering Strategy

To improve predictive signal and capture transaction context before model training, the following feature engineering roadmap is planned:

### 1. Temporal Features (Time-Based)
* **`hour_of_day` (Immediate)**: Extract the hour (0-23) from `timestamp`. Anomalous transactions frequently occur at odd hours (e.g., between 2:00 AM and 4:00 AM).
* **`day_of_week` (Immediate)**: Extract the day of the week (0-6). Helps distinguish weekend vs. weekday spending habits.
* **`is_weekend` (Later)**: Binary flag indicating if the transaction occurred on a weekend.

### 2. Geographical & Channel Features
* **`is_international` (Immediate)**: A binary flag (0 or 1) indicating if the location country is outside the US (e.g., UK, FR, JP, IN, AU). Cross-border transactions present higher anomaly rates.
* **`device_risk_weight` (Later)**: A probability-based risk weight mapped to the transaction channel (Web and Mobile have higher risk profiles compared to POS swipes).

### 3. Customer Transaction Velocity & History
* **`customer_txn_count_30d` (Immediate)**: Cumulative count of transactions per customer to detect high-frequency card testing velocity.
* **`customer_avg_amount_30d` (Immediate)**: Historical average transaction amount for the customer.
* **`amount_ratio_to_avg` (Immediate)**: The ratio of the current transaction amount to the customer's 30-day average. A ratio significantly higher than 1.0 flags anomalous spend scaling.

---

## Batch Inference Pipeline

A dedicated prediction pipeline module (`src/predict.py`) has been created to perform inference on new, unseen transaction datasets. The pipeline is designed to be fully reusable without retraining the model.

It automatically loads the preprocessor pipeline and the best-performing model, runs extensive integrity checks on the input CSV, reconstructs customer transaction history context for accurate rolling velocity metrics, performs inference, maps outputs to risk levels, and exports predictions.

### 1. Schema Validation Constraints
The input transaction CSV is thoroughly validated on arrival. The pipeline checks for:
* **Missing required columns**: The dataset must contain `transaction_id`, `customer_id`, `timestamp`, `amount`, `merchant_category`, `location`, and `device_type`. (The target column `is_anomaly` is optional).
* **Empty files**: Empty files or header-only files raise a validation error.
* **Incorrect data types**: Non-numeric fields in `amount` or unparseable timestamps raise validation errors.
* **Invalid values**: Non-positive transaction values (`amount` <= 0) or null values in key columns raise validation errors.

### 2. Running Predictions
Run batch predictions from the repository root:
```bash
python src/predict.py --input data/raw/transactions.csv --output data/processed/predictions_output.csv
```

### 3. CLI Options
You can customize the inference execution with these optional flags:
* `--input` / `-i` (Required): Path to input transactions CSV.
* `--output` / `-o` (Required): Path to save prediction results CSV.
* `--model` / `-m` (Optional): Specific trained model `.joblib` path (defaults to the automatically resolved `models/best_model.joblib`).
* `--preprocessor` / `-p` (Optional): Specific preprocessor `.joblib` path (defaults to `models/preprocessor.joblib`).
* `--history` (Optional): Path to historical transaction database to align rolling velocity averages (defaults to `data/processed/transactions_clean.csv`).
* `--risk-config` (Optional): Path to custom risk thresholds config (defaults to `config/risk_scoring_config.json`).

### 4. Output Schema
The generated CSV retains the exact row count and order of the input dataset, adding the following prediction indicators:
* `prediction` (int): Model anomaly class prediction (1 for anomaly, 0 for normal).
* `probability_score` (float): Probability score of the transaction being anomalous (0.00 to 1.00).
* `risk_score` (float): Scaled business risk rating from 0.00 to 100.00.
* `risk_level` (string): Risk classification (`Low Risk`, `Medium Risk`, or `High Risk`) mapped using the thresholds defined in the risk config.
