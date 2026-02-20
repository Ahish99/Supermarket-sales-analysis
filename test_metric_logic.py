
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def test_metrics():
    # Mock data
    y_test = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 320, 380, 500]) # Errors: +10, -10, +20, -20, 0
    
    # Expected MAE: (|10| + |-10| + |20| + |-20| + |0|) / 5 = 12.0
    # Expected MSE: (100 + 100 + 400 + 400 + 0) / 5 = 200.0
    # Expected RMSE: sqrt(200) = 14.1421
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test Data:")
    print(f" Actual: {y_test}")
    print(f" Pred:   {y_pred}")
    print(f"Calculated MAE: {mae}")
    print(f"Calculated RMSE: {rmse}")
    
    assert mae >= 0, "MAE should be non-negative"
    assert rmse >= 0, "RMSE should be non-negative"
    assert abs(mae - 12.0) < 0.001, f"Expected MAE 12.0, got {mae}"
    assert abs(rmse - 14.1421) < 0.001, f"Expected RMSE 14.1421, got {rmse}"
    
    print("\nSUCCESS: Metric logic verification passed!")

if __name__ == "__main__":
    test_metrics()
