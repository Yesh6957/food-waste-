import pandas as pd
import numpy as np
import os

def clean_and_engineer_features(input_path='data/raw_inventory.csv', output_path='data/cleaned_inventory.csv'):
    print("Loading raw messy data...")
    df = pd.read_csv(input_path)

    # --- PHASE 1: DATA CLEANING ---
    print("1. Removing duplicates...")
    df = df.drop_duplicates(ignore_index=True)

    print("2. Fixing impossible values (negative quantities)...")
    df['quantity'] = df['quantity'].apply(lambda x: abs(x) if x < 0 else x)

    print("3. Standardizing text and fixing typos...")
    df['category'] = df['category'].str.strip() # Fixes 'Dairy '
    df['category'] = df['category'].replace({'prodce': 'Produce'}) # Fixes typo

    print("4. Handling missing daily_consumption...")
    # Impute missing consumption smartly using the median of that specific ingredient
    df['daily_consumption'] = df['daily_consumption'].fillna(
        df.groupby('ingredient_name')['daily_consumption'].transform('median')
    )
    # Fallback to global median if any remain
    df['daily_consumption'] = df['daily_consumption'].fillna(df['daily_consumption'].median())

    print("5. Handling missing expiry dates...")
    current_date = pd.to_datetime('2026-05-31') 
    df['expiry_date'] = pd.to_datetime(df['expiry_date'], errors='coerce')
    
    # Calculate days to expiry for known dates
    df['days_to_expiry'] = (df['expiry_date'] - current_date).dt.days

    # Impute missing days_to_expiry based on the median shelf-life of its category
    category_expiry_medians = df.groupby('category')['days_to_expiry'].median()
    for category in df['category'].unique():
        mask = (df['category'] == category) & (df['days_to_expiry'].isna())
        df.loc[mask, 'days_to_expiry'] = category_expiry_medians[category]

    # --- PHASE 2: FEATURE ENGINEERING ---
    print("6. Engineering predictive features...")
    
    # Feature A: Stock Duration (How many days the current inventory will last)
    df['stock_duration_days'] = df['quantity'] / df['daily_consumption']
    
    # Feature B: Waste Risk Ratio (If > 1.0, we have more stock than time to use it)
    # Adding 0.1 to avoid division by zero for items expiring today
    df['waste_risk_ratio'] = df['stock_duration_days'] / (df['days_to_expiry'].replace(0, 0.1)) 
    
    # Feature C: The Target Variable (1 = High Risk of Waste, 0 = Safe)
    # An item is at risk if waste_risk_ratio > 1 OR it expires in less than 2 days
    df['is_waste_risk'] = np.where((df['waste_risk_ratio'] > 1.0) | (df['days_to_expiry'] < 2), 1, 0)

    # Save the pipeline output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Cleaned and engineered dataset saved to {output_path}")
    print("\nPreview of engineered features for the ML Model:")
    print(df[['ingredient_name', 'quantity', 'days_to_expiry', 'waste_risk_ratio', 'is_waste_risk']].head())

if __name__ == "__main__":
    clean_and_engineer_features()