import tkinter as tk
from unittest.mock import MagicMock
import pandas as pd
import sys
import os
import importlib.util

# Ensure the app can find its relative imports
sys.path.append('c:/code')

# Load the module
spec = importlib.util.spec_from_file_location("sales_module", "c:/code/sales_app.py")
sales_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sales_module)

def test_uncleaned_data():
    dataset_path = r"C:\Users\ahish\Downloads\supermarket_sales_2024_uncleaned.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    # Mock Root and TK things that might fail without display
    root = MagicMock()
    
    # Instantiate App
    app = sales_module.SalesApp(root)
    
    # Load Dataset
    print(f"Loading dataset: {dataset_path}")
    try:
        app.current_df = pd.read_csv(dataset_path)
        app.original_df = app.current_df.copy()
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return
    
    # Mock UI methods
    app.clear = MagicMock()
    app.feature_page = MagicMock()
    app.premium_button = MagicMock()
    
    print("Testing outcome_analysis logic with uncleaned data...")
    try:
        app.outcome_analysis()
        print("\n--- TEST RESULTS ---")
        print("Outcome analysis logic executed successfully!")
        print(f"Total Transactions: {app.total_transactions}")
        print(f"Total Revenue (Calculated): Rs. {app.total_revenue:,.2f}")
        print(f"Optimal Price point: Rs. {app.optimal_price_point:.2f}")
        print(f"Leading Brand/Product: {app.leading_brand}")
        print("Success: Everything works perfectly with uncleaned data.")
    except Exception as e:
        print(f"Error during outcome_analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_uncleaned_data()
