import pandas as pd
import numpy as np

def verify_logic():
    print("Creating mock data...")
    # Mock Data
    data = {
        'Product': ['Item A', 'Item B', 'Item C', 'Item D'],
        'Category': ['Cat 1', 'Cat 1', 'Cat 2', 'Cat 2'],
        'Quantity': [1000, 10, 500, 50],
        'UnitPrice': [10, 10, 100, 100],
        'Date': pd.date_range(start='2024-01-01', periods=4),
    }
    source_df = pd.DataFrame(data)
    
    print("Testing Logic Block...")
    p_col = 'UnitPrice'
    q_col = 'Quantity'
    i_col = 'Product'
    date_col = 'Date'
    
    # 1. Sales Velocity
    dates = pd.to_datetime(source_df[date_col], errors='coerce')
    days_active = max((dates.max() - dates.min()).days, 1)
    print(f"Days Active: {days_active}")
    
    item_stats = source_df.groupby(i_col).agg({
        q_col: 'sum',
        p_col: 'mean'
    }).reset_index()
    item_stats.columns = ['product', 'total_qty', 'avg_price']
    
    item_stats['velocity'] = item_stats['total_qty'] / days_active
    print("Velocity Header:", item_stats.columns)
    
    avg_velocity_global = item_stats['velocity'].mean()
    avg_price_global = item_stats['avg_price'].mean()
    
    print(f"Global Avg Velocity: {avg_velocity_global}")
    print(f"Global Avg Price: {avg_price_global}")
    
    for _, row in item_stats.iterrows():
        velocity = row['velocity']
        price = row['avg_price']
        
        # Test Inventory Logic
        lead_time = 14
        reorder_point = int(velocity * lead_time)
        
        # Test Pricing Logic
        is_premium = price > avg_price_global
        is_volume_high = velocity > avg_velocity_global
        
        print(f"Item: {row['product']}, Vel: {velocity:.2f}, IsPrem: {is_premium}, IsVolHigh: {is_volume_high}")

    print("Logic verification successful!")

if __name__ == "__main__":
    verify_logic()
