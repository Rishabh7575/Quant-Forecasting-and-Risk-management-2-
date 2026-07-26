"""
Synthetic Financial Transactions Data Generator.

This module generates a synthetic transactions dataset with normal and anomalous/fraudulent
patterns for demonstration and testing of the risk and anomaly detection engine.
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def generate_synthetic_data(num_rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset of financial transactions.
    
    Args:
        num_rows (int): Number of transactions to generate.
        seed (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: DataFrame containing transaction details.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Pre-define some constants for categories
    merchants = ["online_retail", "grocery", "dining", "travel", "cash_withdrawal", "transfer"]
    locations = ["New York, US", "Los Angeles, US", "Chicago, US", "Houston, US", "Miami, US"]
    intl_locations = ["London, UK", "Paris, FR", "Tokyo, JP", "Mumbai, IN", "Sydney, AU"]
    devices = ["mobile", "web", "pos", "atm"]
    
    # Initialize lists to store data
    txn_ids = [f"TXN{str(i).zfill(8)}" for i in range(1, num_rows + 1)]
    cust_ids = [f"CUST{str(random.randint(1001, 1500))}" for _ in range(num_rows)]
    
    # Generate sorted timestamps over the last 30 days
    start_date = datetime.now() - timedelta(days=30)
    timestamps = []
    current_time = start_date
    for _ in range(num_rows):
        # Average increment to cover 30 days with num_rows transactions
        increment = random.randint(10, int(30 * 24 * 3600 / num_rows * 2))
        current_time += timedelta(seconds=increment)
        timestamps.append(current_time)
        
    amounts = []
    merchant_cats = []
    locs = []
    dev_types = []
    is_anomaly = []
    
    for i in range(num_rows):
        timestamp = timestamps[i]
        # 1.5% probability of being an anomaly
        is_fraud = 1 if random.random() < 0.015 else 0
        is_anomaly.append(is_fraud)
        
        if is_fraud:
            # Anomalies have distinct patterns:
            # Pattern A: Extremely high amount (lognormal distribution shifted)
            # Pattern B: Cash withdrawal or transfer of high value
            # Pattern C: International location for US customers
            # Pattern D: High amount transactions at odd hours (e.g. 2 AM - 4 AM)
            anomaly_type = random.choice(["high_amount", "intl_location", "odd_hours_high"])
            
            if anomaly_type == "high_amount":
                amounts.append(round(np.random.uniform(5000, 20000), 2))
                merchant_cats.append(random.choice(["transfer", "online_retail"]))
                locs.append(random.choice(locations))
                dev_types.append(random.choice(["web", "mobile"]))
            elif anomaly_type == "intl_location":
                amounts.append(round(np.random.exponential(500) + 100, 2))
                merchant_cats.append(random.choice(["online_retail", "dining"]))
                locs.append(random.choice(intl_locations))
                dev_types.append(random.choice(["web", "mobile"]))
            else:  # odd_hours_high
                # Force timestamp hour to be between 2 and 4 AM
                new_hour = random.choice([2, 3, 4])
                timestamps[i] = timestamp.replace(hour=new_hour)
                amounts.append(round(np.random.uniform(1500, 8000), 2))
                merchant_cats.append(random.choice(["cash_withdrawal", "transfer"]))
                locs.append(random.choice(locations))
                dev_types.append(random.choice(["atm", "mobile"]))
        else:
            # Normal transactions
            # Mostly small to moderate amounts
            cat = random.choice(merchants)
            merchant_cats.append(cat)
            
            if cat == "grocery":
                amount = np.random.normal(50, 15)
            elif cat == "dining":
                amount = np.random.normal(40, 20)
            elif cat == "online_retail":
                amount = np.random.exponential(150) + 10
            elif cat == "travel":
                amount = np.random.normal(600, 250)
            elif cat == "cash_withdrawal":
                amount = random.choice([20, 40, 60, 100, 200, 300])
            else:  # transfer
                amount = np.random.exponential(300) + 20
                
            amount = max(1.0, round(amount, 2))
            amounts.append(amount)
            
            # Local locations
            locs.append(random.choice(locations))
            
            # Normal device mapping
            if cat == "cash_withdrawal":
                dev_types.append("atm")
            elif cat == "online_retail":
                dev_types.append(random.choice(["web", "mobile"]))
            else:
                dev_types.append(random.choice(devices))
                
    df = pd.DataFrame({
        "transaction_id": txn_ids,
        "customer_id": cust_ids,
        "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "amount": amounts,
        "merchant_category": merchant_cats,
        "location": locs,
        "device_type": dev_types,
        "is_anomaly": is_anomaly
    })
    
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    output_path = os.path.join(raw_data_dir, "transactions.csv")
    print(f"Generating synthetic financial transactions data...")
    df_transactions = generate_synthetic_data(num_rows=10000)
    df_transactions.to_csv(output_path, index=False)
    print(f"Successfully generated 10,000 transactions and saved to {output_path}")
