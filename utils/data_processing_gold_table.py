import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table_label_store(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    try:
        # prepare arguments
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        
        # connect to silver table
        partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = silver_loan_daily_directory + partition_name
        df = spark.read.parquet(filepath)
        print('loaded from:', filepath, 'row count:', df.count())
    
        # get customer at mob
        df = df.filter(col("mob") == mob)
    
        # get label
        df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
        df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))
        
        # select columns to save
        df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")
    
        # save gold table - IRL connect to database to write
        partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = gold_label_store_directory + partition_name
        df.write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        print('saved to:', filepath)

        
        return df
    except:
        print("No record found on Date:", snapshot_date_str)
        return None

def process_labels_gold_table_feature_store(silver_customer_directory, gold_feature_store_directory, spark):
    # --- Load customer silver table ----
    filepath = silver_customer_directory + "silver_customer.parquet"
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    df = df.select('Customer_ID', 'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',  'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Months', 'spending_scale', 'value_payments_scale', 'Credit_Mix_scale', 'is_payment_min_amount_yes', "is_payment_min_amount_no", "debt_to_income_ratio", "income_to_emi_ratio", "emi_to_balance_ratio")

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store.parquet"
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    return df