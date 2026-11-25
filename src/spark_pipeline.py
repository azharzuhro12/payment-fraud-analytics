"""
PySpark Big Data Analytics and Feature Engineering Pipeline.
Demonstrates:
- PySpark DataFrame processing & SQL execution over large transactional datasets
- Feature engineering on big data (balance delta, error balance, cyclical time features)
- PySpark MLlib / Spark Window aggregations (rolling count per merchant, velocity)
- Structured + Unstructured data preprocessing (NLP text features from device remarks)
- Export to Parquet for high-speed ML modeling
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def get_spark_session(app_name: str = "PrimaVistaFraudAnalytics"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def process_and_engineer_features(input_parquet: str = "data/raw/transactions.parquet", output_parquet: str = "data/processed/features.parquet"):
    print(f"Starting PySpark session...")
    spark = get_spark_session()
    
    print(f"Reading dataset from {input_parquet} into Spark DataFrame...")
    df = spark.read.parquet(input_parquet)
    total_count = df.count()
    print(f"Successfully loaded {total_count:,} rows with PySpark!")
    
    # Feature 1: Mathematical balance changes & discrepancy checks
    # In fraud, balance changes often don't equal the transaction amount
    print("Computing balance discrepancy and error features...")
    df = df.withColumn(
        "error_balance_orig", 
        F.round(F.col("newbalance_orig") + F.col("amount") - F.col("oldbalance_orig"), 2)
    ).withColumn(
        "error_balance_dest",
        F.round(F.col("oldbalance_dest") + F.col("amount") - F.col("newbalance_dest"), 2)
    ).withColumn(
        "is_account_emptied",
        F.when((F.col("newbalance_orig") == 0) & (F.col("oldbalance_orig") > 0), 1).otherwise(0)
    ).withColumn(
        "amount_to_oldbalance_ratio",
        F.when(F.col("oldbalance_orig") > 0, F.col("amount") / F.col("oldbalance_orig")).otherwise(999.0)
    )
    
    # Feature 2: Temporal & Cyclical time features (24-hour day cycle)
    df = df.withColumn("hour_of_day", F.col("step") % 24)
    df = df.withColumn("day_of_week", (F.floor(F.col("step") / 24) % 7))
    df = df.withColumn("is_night_transaction", F.when((F.col("hour_of_day") >= 0) & (F.col("hour_of_day") <= 5), 1).otherwise(0))
    
    # Cyclical encoding for hour (sin & cos)
    pi = 3.141592653589793
    df = df.withColumn("hour_sin", F.sin(2 * pi * F.col("hour_of_day") / 24.0))
    df = df.withColumn("hour_cos", F.cos(2 * pi * F.col("hour_of_day") / 24.0))
    
    # Feature 3: Window functions (Big Data Velocity - count of transactions in recent steps per merchant)
    print("Computing Spark Window aggregations (transaction velocity & rolling sums)...")
    mch_window = Window.partitionBy("name_dest").orderBy("step").rangeBetween(-24, 0) # past 24 hours
    
    df = df.withColumn("mch_tx_count_24h", F.count("amount").over(mch_window))
    df = df.withColumn("mch_tx_volume_24h", F.round(F.sum("amount").over(mch_window), 2))
    
    # Feature 4: Unstructured Text Processing (Device telemetry and POS error logs)
    # Extracts keywords indicative of fraud risk from device_remarks
    print("Preprocessing unstructured device logs using Spark SQL regex/keyword scoring...")
    suspicious_patterns = [
        "fallback", "repeat", "spike", "emptied", "anomaly", "flagged"
    ]
    
    text_cond = F.lit(0)
    for kw in suspicious_patterns:
        text_cond = text_cond + F.when(F.lower(F.col("device_remarks")).contains(kw), 1).otherwise(0)
        
    df = df.withColumn("device_risk_flag_count", text_cond)
    
    print(f"Saving engineered features to {output_parquet}...")
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.write.mode("overwrite").parquet(output_parquet)
    
    print("PySpark Feature Engineering complete!")
    spark.stop()

if __name__ == "__main__":
    process_and_engineer_features()

