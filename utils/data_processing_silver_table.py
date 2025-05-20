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

from pyspark.sql.functions import col, when, length, regexp_replace, regexp_extract
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table_lms(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    try:
        # prepare arguments
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        
        # connect to bronze table
        partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
        filepath = bronze_lms_directory + partition_name
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        print('loaded from:', filepath, 'row count:', df.count())
    
   
        # clean data: enforce schema / data type
        # Dictionary specifying columns and their desired datatypes
        column_type_map = {
            "loan_id": StringType(),
            "Customer_ID": StringType(),
            "loan_start_date": DateType(),
            "tenure": IntegerType(),
            "installment_num": IntegerType(),
            "loan_amt": FloatType(),
            "due_amt": FloatType(),
            "paid_amt": FloatType(),
            "overdue_amt": FloatType(),
            "balance": FloatType(),
            "snapshot_date": DateType(),
        }
    
        # cast to their respective data types
        # withColumn(colName: str, col: pyspark.sql.column.Column) 
        for column, new_type in column_type_map.items():
            df = df.withColumn(column, col(column).cast(new_type))
    
        # augment data: add month on book
        df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    
        # augment data: add days past due
        df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
        df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
        df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
    
        # save silver table - IRL connect to database to write
        partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = silver_loan_daily_directory + partition_name
        df.write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        print('saved to:', filepath)
        
        return df
    except:
        print("No record found on Date:", snapshot_date_str)
        return None

def process_silver_table_customer(bronze_attributes_directory, bronze_financials_directory, silver_customer_directory, spark):
    # ----------- Handle Feature Attributes ----------------
    # connect to bronze table - attribute
    filepath_attributes = bronze_attributes_directory + "bronze_attributes.csv"
    df_attributes = spark.read.csv(filepath_attributes, header=True, inferSchema=True)
    print('loaded from:', filepath_attributes, 'row count:', df_attributes.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map_attributes = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    # handle invalid data in 'age' column
    df_attributes = df_attributes.withColumn("age", regexp_replace("age", "_", ""))
    df_attributes = df_attributes.withColumn("age", regexp_replace("age", "-", ""))

    # Name, Occupation, SSN will not be used (dropped) later
    
    # cast to their respective data types
    # withColumn(colName: str, col: pyspark.sql.column.Column) 
    for column, new_type in column_type_map_attributes.items():
        df_attributes = df_attributes.withColumn(column, col(column).cast(new_type))


    
    # ----------- Handle Feature Financials ----------------
    # connect to bronze table - financials
    filepath_financials = bronze_financials_directory + "bronze_financials.csv"
    df_financials = spark.read.csv(filepath_financials, header=True, inferSchema=True)
    print('loaded from:', filepath_financials, 'row count:', df_financials.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map_financials = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    # handle invalid data
    # replace _ with either 0 or nothing (remove underscore)
    col_to_process = ["Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", "Outstanding_Debt", "Amount_invested_monthly"]
    for cp in col_to_process:
        df_financials = df_financials.withColumn(cp, regexp_replace(cp, "_", ""))    
    df_financials = df_financials.withColumn("Changed_Credit_Limit", regexp_replace("Changed_Credit_Limit", "_", "0"))
    

    # Round negatives to 0
    col_to_process = ["Num_Bank_Accounts", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit"]
    for cp in col_to_process:
        df_financials = df_financials.withColumn(cp, when(col(cp) < 0, 0).otherwise(col(cp)))

    # modify column Num_of_Loan from total number of Type_of_Loan
    # if Num_of_Loan == 0... leave it unchanged
    # otherwise, Num_of_Loan = number of commas in Type_of_Loan + 1
    num_commas = length(col("Type_of_Loan")) - length(regexp_replace(col("Type_of_Loan"), ",", ""))
    
    df_financials = df_financials.withColumn(
        "Num_of_Loan",
        when(col("Num_of_Loan") == 0, col("Num_of_Loan")).otherwise(num_commas + 1)
    )

    # total months Credit_History_Age
    df_financials = df_financials.withColumn("years", regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast("int"))
    df_financials = df_financials.withColumn("months", regexp_extract(col("Credit_History_Age"), r"(\d+) Months", 1).cast("int"))
    # total months = years * 12 + months
    df_financials = df_financials.withColumn("Credit_History_Months", col("years") * 12 + col("months"))


    # Payment_Behaviour --> spending_scale and value_payments_scale
    df_financials = df_financials.withColumn( "spending_scale", regexp_extract(col("Payment_Behaviour"), r"(Low|High)_spent", 1) )
    df_financials = df_financials.withColumn( "spending_scale", 
                                             when(col("spending_scale") == "Low", 1).when(col("spending_scale") == "High", 2).otherwise(0) )

    
    df_financials = df_financials.withColumn( "value_payments_scale", regexp_extract(col("Payment_Behaviour"), r"(Small|Medium|Large)_value_payments", 1) )
    df_financials = df_financials.withColumn( "value_payments_scale", 
                                             when(col("value_payments_scale") == "Small", 1).when(col("value_payments_scale") == "Medium", 2).when(col("value_payments_scale") == "Large", 3).otherwise(0) )

    # Credit Mix
    df_financials = df_financials.withColumn( "Credit_Mix_scale", when(col("Credit_Mix") == "Bad", 1).when(col("Credit_Mix") == "Standard", 2).when(col("Credit_Mix") == "Good", 3).otherwise(0) )

    # Payment_of_Min_Amount - one hot encode
    df_financials = df_financials.withColumn( "is_payment_min_amount_yes", when(col("Payment_of_Min_Amount") == "Yes", 1).otherwise(0) ) 
    df_financials = df_financials.withColumn( "is_payment_min_amount_no", when(col("Payment_of_Min_Amount") == "No", 1).otherwise(0) )
    
    
    # cast to their respective data types
    # withColumn(colName: str, col: pyspark.sql.column.Column) 
    for column, new_type in column_type_map_financials.items():
        df_financials = df_financials.withColumn(column, col(column).cast(new_type))

        
    # ----------- Handle Customer Table --> Join Feature Attributes + Financial ----------------
    df_joined = df_attributes.join( df_financials, on=['Customer_ID', 'snapshot_date'], how='left' )
    print(df_joined.count())


    filepath = silver_customer_directory + "silver_customer.parquet"
    df_joined.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    return df_joined
    

    