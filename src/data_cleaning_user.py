# DATA CLEANING
import pandas as pd
import os

# Load raw data
df = pd.read_csv("data/userbehaviour_dataset.csv")

print("Before Cleaning:", df.shape)
print(df.head())

# Check missing values
print(df.isnull().sum())

# Handle missing values
df["price"] = df["price"].fillna(df["price"].median())
df["quantity"] = df["quantity"].fillna(1)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)

# Remove duplicates
df = df.drop_duplicates()

# Add time features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# Save cleaned data
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/cleaned_user_behavior.csv", index=False)

print("Cleaning Done!")