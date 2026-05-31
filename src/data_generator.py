import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_messy_inventory(num_records=1200):
    np.random.seed(42)
    random.seed(42)
    
    categories = {
        'Dairy': ['Milk', 'Paneer', 'Cheese', 'Butter', 'Yogurt', 'Cream'],
        'Produce': ['Tomatoes', 'Onions', 'Spinach', 'Potatoes', 'Carrots', 'Mushroom', 'Bell Peppers'],
        'Meat': ['Chicken Breast', 'Mutton', 'Fish Fillet', 'Prawns', 'Beef'],
        'Dry Pantry': ['Rice', 'Flour', 'Lentils', 'Pasta', 'Sugar', 'Salt', 'Spices']
    }
    
    suppliers = ['AgroFresh', 'Metro Wholesale', 'Local Mandi', 'Prime Meats', 'DairyCo']
    storage_types = ['Refrigerated', 'Ambient', 'Frozen']
    
    data = []
    current_date = datetime(2026, 5, 31) # Setting a static date for consistent testing
    
    for _ in range(num_records):
        category = random.choice(list(categories.keys()))
        ingredient = random.choice(categories[category])
        
        # Base realistic metrics
        quantity = round(random.uniform(1.0, 50.0), 2)
        daily_consumption = round(random.uniform(0.5, 10.0), 2)
        
        # Expiry logic based on category
        if category == 'Dairy':
            days_to_expire = random.randint(-2, 10) # -2 implies already expired
        elif category == 'Produce':
            days_to_expire = random.randint(-1, 14)
        elif category == 'Meat':
            days_to_expire = random.randint(-2, 7)
        else: # Dry Pantry
            days_to_expire = random.randint(30, 365)
            
        expiry_date = (current_date + timedelta(days=days_to_expire)).strftime('%Y-%m-%d')
        
        storage = 'Frozen' if category == 'Meat' else ('Refrigerated' if category in ['Dairy', 'Produce'] else 'Ambient')
        
        record = {
            'ingredient_name': ingredient,
            'quantity': quantity,
            'expiry_date': expiry_date,
            'category': category,
            'daily_consumption': daily_consumption,
            'supplier': random.choice(suppliers),
            'wastage_history': round(random.uniform(0, 5.0), 2), # Historical wastage in kg
            'storage_type': storage
        }
        data.append(record)
        
    df = pd.DataFrame(data)
    
    # --- INTRODUCING INTENTIONAL CHAOS (Real-world simulation) ---
    print("Injecting real-world noise into the dataset...")
    
    # 1. Missing Values (Nulls)
    df.loc[df.sample(frac=0.05).index, 'expiry_date'] = np.nan
    df.loc[df.sample(frac=0.02).index, 'daily_consumption'] = np.nan
    
    # 2. Inconsistent Formats & Typos
    df.loc[df.sample(frac=0.03).index, 'category'] = 'Dairy ' # trailing space
    df.loc[df.sample(frac=0.02).index, 'category'] = 'prodce' # typo
    
    # 3. Impossible values (Negative quantities)
    df.loc[df.sample(frac=0.01).index, 'quantity'] = -5.0
    
    # 4. Duplicate entries
    duplicates = df.sample(frac=0.02)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save the file
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', 'raw_inventory.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Success! Generated {len(df)} messy records and saved to {output_path}")
    print(df.head())

if __name__ == "__main__":
    generate_messy_inventory()