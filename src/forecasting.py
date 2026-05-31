import pandas as pd
import os

def generate_forecast(input_path='data/cleaned_inventory.csv', output_path='reports/demand_forecast.csv'):
    print("Loading inventory data for demand forecasting...")
    df = pd.read_csv(input_path)

    print("Calculating 7-day and 14-day demand projections...")
    # Project future demand based on daily consumption rates
    df['projected_7d_demand'] = df['daily_consumption'] * 7
    df['projected_14d_demand'] = df['daily_consumption'] * 14

    print("Identifying Inventory Anomalies (Shortages & Overstocks)...")
    
    # Shortage Risk: Current quantity won't even last 7 days
    df['is_shortage_risk'] = df['quantity'] < df['projected_7d_demand']
    
    # Overstock Risk: Current quantity exceeds 21 days of demand (ignoring Dry Pantry items which have long shelf lives)
    df['is_overstock_risk'] = (df['quantity'] > (df['daily_consumption'] * 21)) & (df['category'] != 'Dry Pantry')

    # Select the most relevant columns for the final report
    forecast_df = df[['ingredient_name', 'category', 'quantity', 'daily_consumption', 
                      'projected_7d_demand', 'is_shortage_risk', 'is_overstock_risk']]
    
    # Save the forecasting report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecast_df.to_csv(output_path, index=False)

    print(f"\nSuccess! Forecast generated and saved to {output_path}")
    
    # Display a quick summary of the risks found
    shortages = forecast_df['is_shortage_risk'].sum()
    overstocks = forecast_df['is_overstock_risk'].sum()
    print(f"Actionable Insights Found: {shortages} items at risk of shortage, {overstocks} items overstocked.")
    
    print("\nForecast Preview:")
    print(forecast_df.head())

if __name__ == "__main__":
    generate_forecast()