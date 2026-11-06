# ETL PIPELINE

import pandas as pd
from sqlalchemy import create_engine
import re

# DB connection
engine = create_engine("sqlite:///ecommerce.db")


def extract():
    # Load user features created in Week 1
    users = pd.read_parquet(
        "data/processed/user_features.parquet"
    )

    # Load processed product data
    products = pd.read_csv(
        "data/processed/processed_products.csv"
    )

    return users, products


def transform(users, products):

    # Remove rows with missing values
    users.dropna(inplace=True)
    products.drop_duplicates(inplace=True)

    # Standardize column names
    users.columns = users.columns.str.lower()
    products.columns = products.columns.str.lower()

    # Clean column names to remove special characters
    users.columns = [
        re.sub(r'[^a-zA-Z0-9_]', '', col)
        for col in users.columns
    ]

    products.columns = [
        re.sub(r'[^a-zA-Z0-9_]', '', col)
        for col in products.columns
    ]

    return users, products


def load(users, products):

    users.to_sql(
        "users",
        engine,
        if_exists="replace",
        index=False
    )

    products.to_sql(
        "products",
        engine,
        if_exists="replace",
        index=False
    )


def run_etl():

    users, products = extract()

    users, products = transform(
        users,
        products
    )

    print("\nNull values after transformation for users:")
    print(users.isnull().sum())

    print("\nNull values after transformation for products:")
    print(products.isnull().sum())

    load(users, products)

    print("\nETL Pipeline Done!")


if __name__ == "__main__":
    run_etl()