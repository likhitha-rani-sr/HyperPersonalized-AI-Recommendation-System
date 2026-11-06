# FEATURE ENGINEERING
import pandas as pd
import os

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_user_behavior.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ------------------------------------------
# RFM FEATURES
# ------------------------------------------
purchase_df = df[df["event_type"] == "purchase"]

rfm = purchase_df.groupby("user_id").agg({
    "timestamp": lambda x: (df["timestamp"].max() - x.max()).days,
    "user_id": "count",
    "price": "sum"
})

rfm.columns = ["recency", "frequency", "monetary"]

# ------------------------------------------
# USER BEHAVIOR FEATURES
# ------------------------------------------
views = df[df["event_type"] == "view"].groupby("user_id").size()
cart = df[df["event_type"] == "cart"].groupby("user_id").size()
wishlist = df[df["event_type"] == "wishlist"].groupby("user_id").size()

rfm["total_views"] = views
rfm["total_cart"] = cart
rfm["total_wishlist"] = wishlist

rfm = rfm.fillna(0)
rfm.reset_index(inplace=True)

# ------------------------------------------
# SAVE PARQUET
# ------------------------------------------
os.makedirs("data/processed", exist_ok=True)
rfm.to_parquet("data/processed/user_features.parquet", index=False)

print("Feature Engineering Done!")