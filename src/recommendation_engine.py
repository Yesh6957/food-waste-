import pandas as pd
import os
from google import genai

# Fetch the key securely from the environment variables
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("[WARNING] GEMINI_API_KEY environment variable not found. AI features will fail.")

client = genai.Client(api_key=API_KEY)

def load_prompt_template(filepath='prompts/chef_special_prompt.txt'):
    with open(filepath, 'r') as file:
        return file.read()

# ... [KEEP THE REST OF YOUR CODE EXACTLY THE SAME BELOW THIS LINE] ...

def generate_recommendations(data_path='data/cleaned_inventory.csv'):
    print("Loading inventory data to find high-risk items...")
    df = pd.read_csv(data_path)
    
    # Filter for items flagged as high risk
    high_risk_items = df[df['is_waste_risk'] == 1].sort_values(by='days_to_expiry')
    
    if high_risk_items.empty:
        print("No high-risk items found! Kitchen is running perfectly.")
        return
        
    # Get the top 5 most urgent items
    top_urgent = high_risk_items.head(5)
    ingredients_list = ", ".join(top_urgent['ingredient_name'].tolist())
    
    print(f"Urgent ingredients identified: {ingredients_list}")
    
    # Load and format the prompt
    prompt_template = load_prompt_template()
    final_prompt = prompt_template.format(ingredients_list=ingredients_list)
    
    print("\nQuerying Gemini API (2.5-Flash) for Live Chef Specials...\n")
    print("-" * 50)
    
    try:
        # Using the new models.generate_content method with the upgraded model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=final_prompt
        )
        ai_response = response.text
        print(ai_response)
        
    except Exception as e:
        print(f"\n[WARNING] API Call Failed: {e}")
        print("Triggering gracefully degraded fallback response...")
        ai_response = f"Fallback Recommendation: Create a slow-cooked broth or stew using the {ingredients_list} to extend shelf life by 3 days."
        print(ai_response)
        
    print("-" * 50)
    
    # Save the output
    os.makedirs('reports', exist_ok=True)
    with open('reports/latest_recommendations.txt', 'w', encoding='utf-8') as f:
        f.write(ai_response)
        
    print("Success! Live recommendations saved to reports/latest_recommendations.txt")

if __name__ == "__main__":
    generate_recommendations()