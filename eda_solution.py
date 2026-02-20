import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- SCENARIO ---
# You have a dataset with categorical variables ('Color', 'Size').
# You want to do EDA (charts, graphs) on categories, but the model needs numbers.

# 1. Setup Mock Data
data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue'],
    'Size': ['S', 'M', 'L', 'S', 'M', 'L', 'X', 'M'],
    'Price': [10, 20, 30, 10, 20, 30, 15, 25],
    'Target': [0, 1, 1, 0, 1, 1, 0, 1]
})

print("=== 1. RAW DATA (Ideal for EDA) ===")
print(data[['Color', 'Size']].head())
print("\n[EDA Insight]: We can easily plot bar charts of 'Color' because it's still text.")
print("-" * 50)

# 2. THE SOLUTION: Train/Test Split on RAW data
# Don't encode yet! Split first.
X = data.drop('Target', axis=1)
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("\n=== 2. TRAINING DATA (Still Raw) ===")
print("X_train head:\n", X_train.head())
print("\n[EDA Benefit]: X_train preserves 'Red', 'Blue', etc. You can run pandas profiling or seaborn here.")
print("-" * 50)

# 3. THE PIPELINE: Embedding Encoding into the Model
# We define a transformer that handles encoding specifically for the model.
categorical_features = ['Color', 'Size']

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # Passthrough numerical columns like 'Price'
)

# Bundle preprocessor and modeling together
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# 4. TRAIN
# We pass the RAW X_train to the pipeline. The pipeline handles encoding on the fly.
pipeline.fit(X_train, y_train)

print("\n=== 3. MODEL TRAINING ===")
print("Pipeline fitted successfully.")
print("The model 'sees' encoded numbers, but your variable 'X_train' remains human-readable.")

# 5. PREDICTION
# We can even pass raw data for prediction!
prediction = pipeline.predict(pd.DataFrame({'Color': ['Red'], 'Size': ['S'], 'Price': [100]}))
print(f"\nPrediction for (Red, S, 100): {prediction[0]}")
