
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- 1. SETUP MOCK DATA ---
print("\n--- 1. Creating Mock Data ---")
df = pd.DataFrame({
    'Brand': ['Nike', 'Adidas', 'Nike', 'Puma', None], # Categorical with missing
    'Price': [100, 50, 2000, 100, None], # Numeric with outlier (2000) and missing
    'Quantity': [2, 4, 1, 10, 5]
})
print("Original Data:")
print(df)

# --- 2. TEST MISSING VALUES (Mean/Mode) ---
print("\n--- 2. Testing Missing Value Handling (Mean/Mode) ---")
cleaned_df = df.copy()

# A. Numeric Mean
mean_price = cleaned_df['Price'].mean()
print(f"Mean Price: {mean_price}")
cleaned_df['Price'].fillna(mean_price, inplace=True)

# B. Categorical Mode
# Logic in App: Convert to Title Case -> Find Mode -> Fill
cleaned_df['Brand'] = cleaned_df['Brand'].astype(str).str.title()
# Note: 'None' string might become 'None' or 'Nan', app regex handles this usually in separate step,
# but here we test the core fillna logic.
# In app, it uses pd.isnull() check.
# Let's clean the 'None' string back to real NaN for proper mode calc like the app likely expects data state
cleaned_df['Brand'].replace({'None': np.nan, 'Nan': np.nan}, inplace=True)

mode_brand = cleaned_df['Brand'].mode()[0]
print(f"Mode Brand: {mode_brand}")

cleaned_df['Brand'].fillna(mode_brand, inplace=True)

print("Data after Missing Value Handling:")
print(cleaned_df)

if cleaned_df.isnull().sum().sum() == 0:
    print("[PASS] Method: Missing values filled.")
else:
    print("[FAIL] Missing values remain.")
    print(cleaned_df.isnull().sum())

# --- 3. TEST OUTLIER REMOVAL (IQR) ---
print("\n--- 3. Testing Outlier Removal (IQR) ---")
# Reset check for outliers
df_outlier = cleaned_df.copy()
col = 'Price'
Q1 = df_outlier[col].quantile(0.25)
Q3 = df_outlier[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(f"IQR Bounds for {col}: {lower} to {upper}")

df_outlier = df_outlier[(df_outlier[col] >= lower) & (df_outlier[col] <= upper)]
print(f"Data after Outlier Removal (Shape {df_outlier.shape}):")
print(df_outlier)

if 2000 not in df_outlier['Price'].values:
    print("[PASS] Outlier (2000) removed.")
else:
    print("[FAIL] Outlier (2000) NOT removed.")

# --- 4. TEST ENCODING ---
print("\n--- 4. Checking Encoding Logic ---")
df_encode = df_outlier.copy()
le = LabelEncoder()
df_encode['Brand_Encoded'] = le.fit_transform(df_encode['Brand'])
print("Encoded Data:")
print(df_encode[['Brand', 'Brand_Encoded']])

if pd.api.types.is_numeric_dtype(df_encode['Brand_Encoded']):
    print("[PASS] Brand column encoded to numeric.")
else:
    print("[FAIL] Brand column not numeric.")
