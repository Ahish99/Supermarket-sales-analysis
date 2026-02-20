
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os
import tkinter as tk

# Add path
sys.path.append('c:/code')

# Define mock classes/modules before importing sales_app if needed
# But mostly we just want to run the app class method.

class TestSalesAppGUI(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n--- Setting up UI Test Environment ---")
        # Initialize root but dont show it
        cls.root = tk.Tk()
        cls.root.withdraw() # Hide window
        
        # Import App
        from sales_app import SalesApp
        cls.app = SalesApp(cls.root)
        
        # Path to mock data
        cls.csv_path = os.path.abspath('c:/code/mock_sales_data.csv')
        if not os.path.exists(cls.csv_path):
            # Create if missing (fallback)
            data = {
                'Invoice_ID': range(10),
                'Price': [100] * 10,
                'Quantity': [2] * 10,
                'Brand': ['BrandA', 'BrandB'] * 5
            }
            pd.DataFrame(data).to_csv(cls.csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        # Reset page state if needed
        pass

    def test_01_initialization(self):
        print("\n[Test] Initialization")
        self.assertIsNotNone(self.app.root)
        self.assertEqual(self.app.root.title(), "Sales Analysis System")

    @patch('tkinter.filedialog.askopenfilename')
    @patch('tkinter.messagebox.showinfo')
    def test_02_load_data(self, mock_msg, mock_file):
        print("\n[Test] Load Data (Mock)")
        
        # Simulate user selecting the file
        mock_file.return_value = self.csv_path
        
        # Trigger upload logic
        self.app.upload_dataset()
        
        # Verify
        self.assertIsNotNone(self.app.current_df)
        self.assertGreater(len(self.app.current_df), 0)
        print(f"[PASS] Loaded {len(self.app.current_df)} rows from mock data.")

    def test_03_navigation(self):
        print("\n[Test] Navigation")
        # Test Dashboard
        self.app.dashboard()
        self.assertEqual(self.app.current_page, "dashboard")
        
        # Test Feature Page
        self.app.feature_page()
        self.assertEqual(self.app.current_page, "feature")
        
        print("[PASS] Navigation/Page state updates correctly.")

    @patch('tkinter.messagebox.showinfo')
    def test_04_calculation(self, mock_msg):
        print("\n[Test] Total Sale Calculation")
        # Ensure data is loaded
        if self.app.current_df is None:
            self.app.current_df = pd.read_csv(self.csv_path)
            
        # Ensure 'TotalSale' is not yet present (or ignore if it is)
        if 'TotalSale' in self.app.current_df.columns:
            self.app.current_df.drop(columns=['TotalSale'], inplace=True)
            
        self.app.calculate_total_sale()
        
        self.assertIn('TotalSale', self.app.current_df.columns)
        # Check Value: Price * Quantity
        # Row 0: 60 * 2 = 120 (based on mock data view)
        first_row = self.app.current_df.iloc[0]
        expected = first_row['Price'] * first_row['Quantity']
        self.assertAlmostEqual(first_row['TotalSale'], expected)
        print("[PASS] TotalSale calculated correctly.")

    @patch('tkinter.messagebox.showinfo')
    @patch('tkinter.messagebox.askyesno', return_value=True) 
    def test_05_train_model_logic(self, mock_ask, mock_msg):
        print("\n[Test] Model Training Logic")
        
        if self.app.current_df is None:
            self.app.current_df = pd.read_csv(self.csv_path)

        # Mock UI dependencies normally set up by model_training() page
        self.app.target_var = tk.StringVar()
        self.app.log_text = MagicMock()
        
        # Mock train_models to avoid further deep logic execution that might require more UI
        self.app.train_models = MagicMock()

        try:
            self.app.train_sales_prediction_model()
            
            # Verify Flow
            # 1. TotalSale calculation/check
            self.assertIn('TotalSale', self.app.current_df.columns, "TotalSale should be ensured")
            # 2. Target set
            self.assertEqual(self.app.target_var.get(), 'TotalSale')
            # 3. train_models called
            self.app.train_models.assert_called_once()
            
            print("[PASS] Model training setup logic executed correctly.")
        except Exception as e:
             self.fail(f"Model training logic crashed: {e}")

if __name__ == '__main__':
    unittest.main()
