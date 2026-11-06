# PRODUCT DATA PROCESSING

import pandas as pd
import os

# Load product data
df = pd.read_csv("data/products_dataset.csv")

print("Before Processing:", df.shape)
print(df.head())

# Check missing values
print(df.isnull().sum())

# Handle missing values
df = df.fillna("Unknown")

# Remove duplicates
df = df.drop_duplicates()

# Save processed product data
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/processed_products.csv", index=False)

print("Product Processing Done!")