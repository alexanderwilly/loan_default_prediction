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


def process_bronze_table_lms(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    row_count = df.count()
    # Filter same as df = df[df['col'] == True]
    print(snapshot_date_str + 'row count:', row_count)

    if row_count > 0:
        # save bronze table to datamart - IRL connect to database to write
        partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
        filepath = bronze_lms_directory + partition_name
        df.toPandas().to_csv(filepath, index=False)
        print('saved to:', filepath)

    return df



def process_bronze_table_attributes(bronze_attributes_directory, spark):
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/features_attributes.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    row_count = df.count()
    # Filter same as df = df[df['col'] == True]
    print('row count:', row_count)

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_attributes.csv"
    filepath = bronze_attributes_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df


def process_bronze_table_financials(bronze_financials_directory, spark):
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/features_financials.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    row_count = df.count()
    # Filter same as df = df[df['col'] == True]
    print('row count:', row_count)

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_financials.csv"
    filepath = bronze_financials_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df
