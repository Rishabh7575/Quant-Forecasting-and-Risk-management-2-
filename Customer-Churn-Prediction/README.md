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
