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
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table
print('Done Importing!')

## ---------- Set up PySpark session ------------------
# Initialize Spark Session
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()
# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")
print('Done Initialize Spark Session!')


## ----------- Set up Config --------------------
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2025-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print("Done generate list of dates!")





## ------------ Build Bronze Table ------------------
### ~~~ LMS ~~~
# create bronze datalake
# create multiple csv files 
bronze_lms_directory = "datamart/bronze/lms/"
if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table_lms(date_str, bronze_lms_directory, spark)
print('Done Build Bronze Table - LMS!')

### ~~~ Features Attributes ~~~
# create bronze datalake
bronze_attributes_directory = "datamart/bronze/attributes/"
if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)

# No need to be partitioned
utils.data_processing_bronze_table.process_bronze_table_attributes(bronze_attributes_directory, spark)
print('Done Build Bronze Table - Features Attributes!')


### ~~~ Features Financials ~~~
# create bronze datalake
bronze_financials_directory = "datamart/bronze/financials/"
if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)
    
# No need to be partitioned
utils.data_processing_bronze_table.process_bronze_table_financials(bronze_financials_directory, spark)
print('Done Build Bronze Table - Features Financials!')



## --------- Build Silver Table --------------
### ~~~ LMS ~~~
# create silver datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"
if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table_lms(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
print('Done Build Silver Table - LMS!')



### ~~~ Features Attributes + Financials (Customer) ~~~
# create silver datalake
silver_customer_directory = "datamart/silver/customer/"
if not os.path.exists(silver_customer_directory):
    os.makedirs(silver_customer_directory)

utils.data_processing_silver_table.process_silver_table_customer(bronze_attributes_directory, bronze_financials_directory, silver_customer_directory, spark)
print('Done Build Silver Table - Customer!')


## --------- Build Gold Table --------------
### ~~~ Label Store ~~~
# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"
if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table_label_store(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)
print('Done Build Gold Table - Label Store!')



### ~~~ Feature Store ~~~
# create gold datalake
gold_feature_store_directory = "datamart/gold/feature_store/"
if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)
utils.data_processing_gold_table.process_labels_gold_table_feature_store(silver_customer_directory, gold_feature_store_directory, spark)
print('Done Build Gold Table - Feature Store!')

