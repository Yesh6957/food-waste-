import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_wastage_model(input_path='data/cleaned_inventory.csv', model_dir='models'):
    print("Loading cleaned dataset...")
    df = pd.read_csv(input_path)

    # 1. Select Features and Target
    # We drop 'ingredient_name' and 'supplier' as they are too high-cardinality for this lightweight model,
    # and 'expiry_date' because we already extracted 'days_to_expiry'
    features = ['quantity', 'category', 'daily_consumption', 'wastage_history', 
                'storage_type', 'days_to_expiry', 'stock_duration_days', 'waste_risk_ratio']
    
    X = df[features].copy()
    y = df['is_waste_risk']

    # 2. Encode Categorical Variables
    print("Encoding categorical features...")
    label_encoders = {}
    for col in ['category', 'storage_type']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Train the XGBoost Model
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # 5. Evaluate the Model
    print("\nEvaluating Model Performance...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save the Model and Encoders
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgboost_waste_model.joblib')
    encoders_path = os.path.join(model_dir, 'label_encoders.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(label_encoders, encoders_path)
    
    print(f"Success! Model saved to {model_path}")

if __name__ == "__main__":
    train_wastage_model()