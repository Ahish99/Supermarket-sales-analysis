import pandas as pd
import numpy as np
import random
import os

# Configuration
INPUT_PATH = r"C:\Users\ahish\Downloads\supermarket_sales_2024_uncleaned.csv"
OUTPUT_PATH = r"C:\Users\ahish\Downloads\supermarket_sales_2024_cleaned_proper.csv"

def clean_data():
    print(f"--- Dataset Cleaning Utility ---")
    print(f"Loading dataset: {INPUT_PATH}")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Could not find the file at {INPUT_PATH}")
        return

    try:
        # 1. Load Data
        df = pd.read_csv(INPUT_PATH)
        print(f"Original Shape: {df.shape}")
        
        # 2. Column Name Cleanup (Replicating project's upload logic)
        # Strips whitespace and ensures names are compatible with the modeling pipeline
        df.columns = df.columns.str.strip()
        df.columns = [col.replace(' ', '_').replace('.', '_') for col in df.columns]
        print(f"Step 1: Column names standardized.")

        # 3. Automatic Random Time Imputation (Project-specific fix)
        # Detects 'Time' columns and fills NaNs with HH:MM format
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        for col in time_cols:
            mask = df[col].isna()
            if mask.any():
                print(f"Step 2: Filling {mask.sum()} missing values in '{col}' with random HH:MM...")
                df.loc[mask, col] = [f"{random.randint(0, 23):02}:{random.randint(0, 59):02}" for _ in range(mask.sum())]

        # 4. Missing Value Imputation (Proper Cleaning)
        # Numerical: Mean-based filling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                print(f"Step 3: Filling missing numeric values in '{col}' with mean: {mean_val:.2f}")
                df[col].fillna(mean_val, inplace=True)
                
        # Categorical: Mode-based filling
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    print(f"Step 4: Filling missing categorical values in '{col}' with mode: {mode_val[0]}")
                    df[col].fillna(mode_val[0], inplace=True)

        # 5. Brand Formatting (Replicating project's title case logic)
        brand_cols = [col for col in df.columns if 'brand' in col.lower()]
        for col in brand_cols:
            print(f"Step 5: Formatting brand names in '{col}' to Title Case.")
            df[col] = df[col].astype(str).str.title()

        # 6. Export Cleaned Dataset
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSUCCESS: Cleaned dataset saved to:")
        print(f"-> {OUTPUT_PATH}")
        print(f"Final Count of Missing Values: {df.isnull().sum().sum()}")
        print(f"-----------------------------------")

    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    clean_data()
