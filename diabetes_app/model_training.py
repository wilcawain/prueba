import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():
    print("Loading data...")
    # Load the dataset
    url = 'https://raw.githubusercontent.com/LuisPerezTimana/Webinars/main/diabetes.csv'
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Basic preprocessing (Handling zeros can be added here if strictly required, 
    # but for a basic random forest, we will proceed with the raw numeric values as commonly done in tutorials)
    
    # Feature selection
    # Using all features available in the standard diabetes dataset
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    print("Splitting data...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Standardizing features...")
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest...")
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save artifacts
    print("Saving model and scaler...")
    joblib.dump(rf_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Training complete. Artifacts saved to current directory.")

if __name__ == "__main__":
    train_model()
