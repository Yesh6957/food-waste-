import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai

app = FastAPI(
    title="Yesh Waste Manage AI",
    description="Cinematic, production-grade endpoints for predicting kitchen waste risks."
)

# File Paths
DATA_PATH = 'data/cleaned_inventory.csv'
RAW_DATA_PATH = 'data/raw_inventory.csv'
FORECAST_PATH = 'reports/demand_forecast.csv'

# Securely fetch API key from Render's Environment Variables
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None

class InventoryItem(BaseModel):
    ingredient_name: str
    quantity: float
    category: str
    expiry_date: str

@app.get("/")
def home():
    # Render uses Linux, which is strictly case-sensitive. 
    # Ensures it finds "index.html" perfectly.
    if not os.path.exists("index.html"):
        return {"error": "index.html is missing from the root directory on GitHub!"}
    return FileResponse("index.html")

@app.get("/inventory/high-risk")
def get_high_risk_items():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Dataset missing.")
    df = pd.read_csv(DATA_PATH)
    high_risk_df = df[df['is_waste_risk'] == 1].sort_values(by='days_to_expiry').head(20)
    return high_risk_df[['ingredient_name', 'category', 'quantity', 'days_to_expiry', 'waste_risk_ratio']].to_dict(orient='records')

@app.get("/inventory/chart-data")
def get_chart_data():
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Dataset missing.")
    df = pd.read_csv(DATA_PATH)
    high_risk = df[df['is_waste_risk'] == 1]
    grouped = high_risk.groupby('days_to_expiry')['quantity'].sum().sort_index().head(10)
    labels = [f"{int(day)} Days" for day in grouped.index]
    values = grouped.values.tolist()
    return {"labels": labels, "values": values}

@app.get("/inventory/anomalies")
def get_anomalies():
    if not os.path.exists(FORECAST_PATH):
        raise HTTPException(status_code=404, detail="Forecast missing.")
    df = pd.read_csv(FORECAST_PATH)
    shortages = df[df['is_shortage_risk'] == True].sort_values(by='quantity').head(6)
    overstocks = df[df['is_overstock_risk'] == True].sort_values(by='quantity', ascending=False).head(6)
    return {
        "shortages": shortages[['ingredient_name', 'quantity', 'projected_7d_demand']].to_dict(orient='records'),
        "overstocks": overstocks[['ingredient_name', 'quantity']].to_dict(orient='records')
    }

@app.post("/inventory/add")
def add_inventory_item(item: InventoryItem):
    if not os.path.exists(RAW_DATA_PATH):
        raise HTTPException(status_code=404, detail="Raw dataset missing.")
    new_row = pd.DataFrame([{
        'ingredient_name': item.ingredient_name,
        'quantity': item.quantity,
        'expiry_date': item.expiry_date,
        'category': item.category,
        'daily_consumption': 1.0,
        'supplier': 'Manual Entry',
        'wastage_history': 0.0,
        'storage_type': 'Ambient'
    }])
    new_row.to_csv(RAW_DATA_PATH, mode='a', header=False, index=False)
    return {"message": "Item added successfully."}

@app.post("/recommendations/generate")
def get_chef_specials():
    if not client:
        raise HTTPException(status_code=500, detail="Gemini API Key missing on server.")
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Dataset missing.")
    
    df = pd.read_csv(DATA_PATH)
    high_risk_items = df[df['is_waste_risk'] == 1].sort_values(by='days_to_expiry')
    if high_risk_items.empty:
        return {"message": "All inventory stable."}
        
    top_urgent = high_risk_items.head(5)
    ingredients_list = ", ".join(top_urgent['ingredient_name'].tolist())
    
    prompt = f"""You are an expert Head Chef and inventory optimization specialist.
I have the following ingredients at high risk of expiring soon: {ingredients_list}.
Provide 2 "Chef Special" recipes utilizing these to minimize waste. 
Provide: 1. Dish Name 2. How it utilizes the ingredients 3. Operational reasoning."""
    
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return {"ingredients": ingredients_list, "markdown": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))