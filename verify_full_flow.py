
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. SETUP MOCK DATA ---
print("\n--- 1. Creating Mock Data ---")
# --- 1. SETUP MOCK DATA ---
print("\n--- 1. Creating Mock Data ---")
np.random.seed(42)
n_samples = 200
data = {
    'Brand': np.random.choice(['Nike', 'Adidas', 'Puma'], n_samples),
    'Product': np.random.choice(['Shoe', 'T-Shirt', 'Shorts', 'Sock', 'Hat'], n_samples),
    'Price': np.random.uniform(10, 200, n_samples),
    'Quantity': np.random.randint(1, 20, n_samples),
    'Discount': np.random.choice([0, 0.1, 0.2], n_samples),
    'Noise': np.random.normal(0, 5, n_samples) # Add some noise
}
df = pd.DataFrame(data)
# Linear relationship for Target
df['Total_Amount'] = (df['Price'] * df['Quantity'] * (1 - df['Discount'])) + df['Noise']

# Add some explicitly duplicate rows to test removal
df = pd.concat([df, df.iloc[:5]], ignore_index=True)
print(f"Initial Shape: {df.shape}")

# --- 2. EDA PROCESSING (Simulated) ---
print("\n--- 2. Simulating EDA Processing ---")
# Drop Duplicates
df.drop_duplicates(inplace=True)
print(f"Shape after drop_duplicates: {df.shape}")
# We added 5 duplicates to 200 unique rows, so expect 200
if df.shape[0] != 200:
    print(f"[FAIL] Expected 200 rows, got {df.shape[0]}")
else:
    print("[PASS] Duplicate removal works")

# --- 3. PREPROCESSING (Encoding) ---
print("\n--- 3. Simulating Preprocessing (Encoding) ---")
encoded_df = df.copy()
encoders = {}
for col in encoded_df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    encoders[col] = le

print("Encoded Data Sample:")
print(encoded_df.head())

# --- 4. MODEL TRAINING ---
print("\n--- 4. Simulating Model Training ---")
target = 'Total_Amount'
X = encoded_df.drop(columns=[target])
y = encoded_df[target]

# Feature Selection (Simulating App Logic: Select Numeric)
X_numeric = X.select_dtypes(include=[np.number])
print(f"Features used: {list(X_numeric.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_lr = np.maximum(0, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression R2: {r2_lr:.4f}")

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
y_pred_gb = np.maximum(0, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print(f"Gradient Boosting R2: {r2_gb:.4f}")

if r2_lr > 0.9 and r2_gb > 0.9:
    print("[PASS] Models are predicting correctly (High R2 on deterministic data)")
else:
    print("[WARN] R2 scores seem low for deterministic synthetic data")

# --- 5. OUTCOME ANALYSIS (Leading Brand Check) ---
print("\n--- 5. Simulating Outcome Analysis (Leading Brand) ---")
# Logic from sales_app.py:
# source_df = original_df if available else current_df
# We use 'df' as original and 'encoded_df' as current
source_df = df 

leading_brand = "N/A"
# Prioritize 'brand' keyword specificially
brand_col = next((c for c in source_df.columns if 'brand' in c.lower()), None)
print(f"Found Brand Column: {brand_col}")

# Fallback check (should not trigger here given we have 'Brand')
if not brand_col:
    keywords = ['product', 'item', 'name']
    brand_col = next((c for c in source_df.columns if any(k in c.lower() for k in keywords) and 'id' not in c.lower() and 'code' not in c.lower()), None)

if brand_col:
    top = source_df[brand_col].mode()
    if not top.empty:
        leading_brand = str(top[0])

print(f"Detected Leading Brand: {leading_brand}")

# Test Case 2: What if 'Brand' column is missing but 'Product' exists?
print("\n--- 5b. Testing Fallback Logic (No 'Brand' column) ---")
df_no_brand = df.drop(columns=['Brand'])
brand_col_fb = next((c for c in df_no_brand.columns if 'brand' in c.lower()), None)
if not brand_col_fb:
    keywords = ['product', 'item', 'name']
    brand_col_fb = next((c for c in df_no_brand.columns if any(k in c.lower() for k in keywords) and 'id' not in c.lower() and 'code' not in c.lower()), None)
print(f"Fallback Column Selected: {brand_col_fb}")
    
