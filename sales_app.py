import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
import pandas as pd
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import warnings

# set random seed for reproducibility
random.seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')

# ReportLab Imports for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ================= THEME =================
BLACK = "#050505"
GOLD = "#D4AF37"
CARD = "#0f0f0f"
GREEN = "#28a745"
RED = "#dc3545"
BLUE = "#007bff"
PURPLE = "#6f42c1"
ORANGE = "#fd7e14"
CYAN = "#17a2b8"
PINK = "#e83e8c"
TEAL = "#20c997"
YELLOW = "#ffc107"
GRAY = "#6c757d"
WHITE = "#FFFFFF"
TEXT_DIM = "#888888"

# New color palette for visualizations
COLOR_PALETTE = [GOLD, BLUE, GREEN, RED, PURPLE, ORANGE, CYAN, PINK, TEAL, "#6610f2", "#d63384"]

FONT_TITLE = ("Times New Roman", 28, "bold")
FONT_SUB = ("Times New Roman", 14)
FONT_BTN = ("Times New Roman", 12, "bold")
FONT_SMALL = ("Times New Roman", 10)
FONT_CODE = ("Times New Roman", 11)

# ================= APP =================
class SalesApp:

    def resource_path(self, relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def __init__(self, root):
        self.root = root
        self.root.title("Sales Analysis System")
        self.root.geometry("1200x700")
        self.root.configure(bg=BLACK)

        # Original images for resizing (no blur)
        try:
            self.dashboard_bg_original = Image.open(self.resource_path("SM 3 Bg.jpeg"))
            self.feature_bg_original = Image.open(self.resource_path("SM 2 Bg.jpeg"))
        except:
            self.dashboard_bg_original = None
            self.feature_bg_original = None

        self.bg_img = None
        self.bg_label = None

        self.feature_bg = None
        self.feature_bg_label = None

        self.logo_img = None

        self.recent_files = []
        self.current_df = None
        self.cleaned_df = None # Store cleaned but unencoded data for EDA
        self.encoded_df = None # Store encoded data for Model Training
        self.original_df = None  # Store original data for comparison
        self.current_file = None
        self.dataset_tree = None
        self.dataset_window = None
        self.preprocessing_steps = []  # Track preprocessing steps
        self.brand_encoding_mapping = {}  # Store brand encoding mappings
        self.sidebar_buttons = {}  # Store sidebar button references
        self.current_preprocessing_section = None  # Track current section
        self.eda_window = None  # Track EDA window
        self.model_results = {} # Store all trained models results: key="{target} - {algo}"
        
        # Store original values for visualization
        self.original_brand_names = {}
        self.original_product_names = {}
        self.original_payment_methods = {}
        self.original_stock_status = {}
        
        # track page for resize handling
        self.current_page = "login"
        self.setup_styles()
        self.root.bind("<Configure>", self._on_root_resize)

        self.login_page()

    # ============== NEW: PROJECT SPECIFIC FUNCTIONS ==============
    def calculate_total_sale(self):
        """Calculate TotalSale from UnitPrice * Quantity if columns exist"""
        if self.current_df is not None:
            # Look for price and quantity columns
            price_col = next((c for c in self.current_df.columns 
                            if 'price' in c.lower() or 'unitprice' in c.lower()), None)
            qty_col = next((c for c in self.current_df.columns 
                          if 'quantity' in c.lower() or 'qty' in c.lower()), None)
            
            if price_col and qty_col:
                try:
                    # Convert to numeric
                    price = pd.to_numeric(self.current_df[price_col], errors='coerce')
                    qty = pd.to_numeric(self.current_df[qty_col], errors='coerce')
                    
                    # Calculate TotalSale
                    self.current_df['TotalSale'] = price * qty
                    
                    # ===== FIX: Also add to encoded_df if it exists =====
                    if self.encoded_df is not None:
                        self.encoded_df['TotalSale'] = self.current_df['TotalSale']
                    
                    # Add to preprocessing steps
                    self.preprocessing_steps.append("Calculated TotalSale = UnitPrice × Quantity")
                    messagebox.showinfo("Success", "TotalSale calculated successfully from UnitPrice × Quantity!")
                    return True
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to calculate TotalSale: {str(e)}")
            else:
                messagebox.showwarning("Warning", "Could not find UnitPrice and Quantity columns to calculate TotalSale")
        return False

    def project_specific_insights(self, target_df):
        """Generate insights specifically for supermarket sales project"""
        insights = []
        
        # Check if TotalSale exists, if not try to calculate it
        if 'TotalSale' not in target_df.columns:
            # Try to find price and quantity columns
            price_col = next((c for c in target_df.columns if 'price' in c.lower()), None)
            qty_col = next((c for c in target_df.columns if 'quantity' in c.lower()), None)
            if price_col and qty_col:
                target_df['TotalSale'] = pd.to_numeric(target_df[price_col], errors='coerce') * pd.to_numeric(target_df[qty_col], errors='coerce')
        
        # 1. Best-selling product categories
        category_col = next((c for c in target_df.columns if 'category' in c.lower()), None)
        if category_col:
            top_categories = target_df[category_col].value_counts().head(5)
            insights.append(f"Top 5 Categories: {', '.join(top_categories.index.tolist())}")
        
        # 2. Best-performing months
        date_col = next((c for c in target_df.columns if 'date' in c.lower()), None)
        if date_col and 'TotalSale' in target_df.columns:
            try:
                target_df['Month'] = pd.to_datetime(target_df[date_col], errors='coerce').dt.month_name()
                monthly_sales = target_df.groupby('Month')['TotalSale'].sum()
                if not monthly_sales.empty:
                    best_month = monthly_sales.idxmax()
                    insights.append(f"Best Month: {best_month} (₹{monthly_sales.max():,.2f})")
            except:
                pass
        
        # 3. Payment method preferences
        payment_col = next((c for c in target_df.columns if 'payment' in c.lower()), None)
        if payment_col:
            payment_dist = target_df[payment_col].value_counts(normalize=True)
            if not payment_dist.empty:
                top_payment = payment_dist.index[0]
                insights.append(f"Preferred Payment: {top_payment} ({payment_dist.iloc[0]:.1%})")
        
        # 4. Price vs Quantity relationship
        price_col = next((c for c in target_df.columns if 'price' in c.lower()), None)
        qty_col = next((c for c in target_df.columns if 'quantity' in c.lower()), None)
        if price_col and qty_col:
            correlation = target_df[price_col].corr(target_df[qty_col])
            insights.append(f"Price-Quantity Correlation: {correlation:.3f}")
        
        # 5. Stock clearance candidates (lowest selling)
        product_col = next((c for c in target_df.columns if 'product' in c.lower()), None)
        if product_col:
            low_sellers = target_df[product_col].value_counts().tail(5)
            if not low_sellers.empty:
                insights.append(f"Lowest Sellers: {', '.join(low_sellers.index.tolist()[:3])}")
        
        return insights

    def train_sales_prediction_model(self):
        """Specialized model training for TotalSale prediction"""
        if self.current_df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        # Ensure TotalSale exists
        if 'TotalSale' not in self.current_df.columns:
            if not self.calculate_total_sale():
                messagebox.showerror("Error", "Cannot calculate TotalSale. Need UnitPrice and Quantity columns.")
                return
        
        # ===== FIX: Sync TotalSale to encoded_df if it exists =====
        if self.encoded_df is not None and 'TotalSale' not in self.encoded_df.columns:
            # Add TotalSale to encoded_df
            self.encoded_df['TotalSale'] = self.current_df['TotalSale']
            messagebox.showinfo("Info", "TotalSale added to encoded data for model training")
        
        # Use TotalSale as target
        self.target_var.set('TotalSale')
        
        # Log the prediction task
        self.log_text.insert(tk.END, "\n" + "="*60 + "\n")
        self.log_text.insert(tk.END, "SPECIALIZED: TotalSale Prediction Model\n")
        self.log_text.insert(tk.END, "="*60 + "\n")
        
        # Train models
        self.train_models(clear_display=False)

    def generate_inventory_recommendations(self):
        """Generate inventory planning recommendations based on sales data"""
        if self.current_df is None:
            return []
        
        recommendations = []
        
        # Find necessary columns
        product_col = next((c for c in self.current_df.columns if 'product' in c.lower()), None)
        qty_col = next((c for c in self.current_df.columns if 'quantity' in c.lower()), None)
        date_col = next((c for c in self.current_df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
        
        if product_col and qty_col:
            # Calculate duration
            days_active = 30 # Default fallback
            if date_col:
                try:
                    dates = pd.to_datetime(self.current_df[date_col], errors='coerce').dropna()
                    if not dates.empty:
                        duration = (dates.max() - dates.min()).days + 1 # Use inclusive day count
                        days_active = max(1, duration) # Avoid division by zero
                except:
                    pass

            # Calculate daily sales per product
            product_sales = self.current_df.groupby(product_col)[qty_col].sum()
            avg_daily_sales = product_sales / days_active
            
            # Identify fast vs slow movers based on velocity
            fast_movers = product_sales.nlargest(5)
            slow_movers = product_sales.nsmallest(5)
            
            recommendations.append(f"HIGH DEMAND PRODUCTS (Top Sellers over {days_active} days):")
            for product, sales in fast_movers.items():
                avg_daily = avg_daily_sales.loc[product]
                product_str = str(product)[:25]
                # Suggest reorder point based on 7-day lead time
                reorder_point = int(avg_daily * 7)
                recommendations.append(f"  - {product_str}: Selling {avg_daily:.1f} /day -> Keep min {reorder_point} in stock")
            
            recommendations.append("\nLOW DEMAND PRODUCTS (Review Pricing/Clearance):")
            for product, sales in slow_movers.items():
                product_str = str(product)[:25]
                recommendations.append(f"  - {product_str}: Total {sales} sold in period")
        
        return recommendations

    def clear_model_state(self):
        """Clear model state when data is reset"""
        self.model_results = {}
        if hasattr(self, 'latest_model_metrics'):
            self.latest_model_metrics = None

    # ============== COMMON ==============
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure vertical scrollbar for dark theme
        style.configure("Dark.Vertical.TScrollbar",
                        background=BLACK,
                        darkcolor=BLACK,
                        lightcolor=BLACK,
                        troughcolor=BLACK,
                        bordercolor=BLACK,
                        arrowcolor=GOLD,
                        gripcount=0)
        
        style.map("Dark.Vertical.TScrollbar",
                  background=[('pressed', '#333333'), ('active', '#222222')],
                  arrowcolor=[('pressed', '#FFFFFF'), ('active', '#FFD700')])

    def clear(self):
        """Clears the screen before loading a new page"""
        self.root.configure(bg=BLACK) # Force black background before destroy
        for w in self.root.winfo_children():
            w.destroy()
        self.root.update_idletasks() # Ensure the black bg is painted

    def image_button(self, parent, image_path, cmd):
        """Fits image buttons exactly without any rectangular black border"""
        try:
            # Keep your exact resize logic
            img = Image.open(image_path).resize((240, 70), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Hover Effect: Brighten the image
            enhancer = ImageEnhance.Brightness(img)
            img_hover = enhancer.enhance(1.2)
            photo_hover = ImageTk.PhotoImage(img_hover)
            
            btn = tk.Button(
                parent, 
                image=photo, 
                command=cmd,
                # REMOVED width/height to stop forcing a black box
                bg=BLACK,              # Matches your theme
                activebackground=BLACK,
                bd=0,                  # REMOVES the rectangle border
                highlightthickness=0,  # REMOVES the focus ring
                padx=0, pady=0,        
                relief="flat",         
                cursor="hand2"         #
            )
            btn.image = photo          # Prevents image from disappearing
            btn.image_hover = photo_hover # Keep ref
            
            btn.bind("<Enter>", lambda e: btn.config(image=photo_hover))
            btn.bind("<Leave>", lambda e: btn.config(image=photo))
            return btn
        except Exception as e:
            return tk.Button(parent, text=f"Missing {image_path}", command=cmd,
                           bg=GOLD, fg=BLACK, font=FONT_BTN, relief="raised")
    
    def premium_button(self, parent, text, cmd, width=22, height=2, bg=GOLD, fg=BLACK):
        # Precise replica: Gold body with Black text (default)
        btn = tk.Button(
            parent, 
            text=text, 
            command=cmd,
            font=FONT_BTN, 
            bg=bg,                 
            fg=fg,                 
            bd=2,                  
            relief="raised",       
            cursor="hand2",        
            width=width, 
            height=height,
            activebackground="#FFD700" if bg == GOLD else "#444", 
            activeforeground=BLACK if fg == BLACK else "white"
        )
        
        # Hover effect: slightly lighter gold (only if gold)
        if bg == GOLD:
            btn.bind("<Enter>", lambda e: btn.config(bg="#FFC125")) 
            btn.bind("<Leave>", lambda e: btn.config(bg=GOLD))
        else:
            # Dark button hover
             btn.bind("<Enter>", lambda e: btn.config(bg="#444")) 
             btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn
    
    def create_circle_logo(self):
        try:
            # Make logo larger (increased from 220 to 280)
            logo = Image.open("logo new.png").convert("RGBA").resize((450, 450))
        
            # Remove background by detecting and making the most common border color transparent
            width, height = logo.size
            border_pixels = []
        
            # Get border pixels
            for x in range(width):
                border_pixels.append(logo.getpixel((x, 0)))
            for x in range(width):
                border_pixels.append(logo.getpixel((x, height-1)))
            for y in range(height):
                border_pixels.append(logo.getpixel((0, y)))
            for y in range(height):
                border_pixels.append(logo.getpixel((width-1, y)))
        
            from collections import Counter
            color_counter = Counter(border_pixels)
            if color_counter:
                background_color = color_counter.most_common(1)[0][0]
                datas = logo.getdata()
                new_data = []
                for item in datas:
                    if (abs(item[0] - background_color[0]) < 30 and 
                        abs(item[1] - background_color[1]) < 30 and 
                        abs(item[2] - background_color[2]) < 30):
                        new_data.append((item[0], item[1], item[2], 0))
                    else:
                        new_data.append(item)
                logo.putdata(new_data)
        
            self.logo_img = ImageTk.PhotoImage(logo)
            return self.logo_img
        except:
            return None

    def _on_root_resize(self, event):
        if self.current_page == "dashboard" and self.bg_label is not None:
            self._resize_dashboard_bg()
        elif self.current_page == "feature":
            # 1. Resize the background
            self._resize_feature_bg()
            
            # 2. ADD THIS: Keep the buttons in the center when maximizing
            for child in self.root.winfo_children():
                if isinstance(child, tk.Canvas):
                    try:
                        child.coords(self.feat_win, event.width // 2, event.height // 2)
                    except:
                        pass

    # ============== LOGIN ==============
    def login_page(self):
        self.current_page = "login"
        self.clear()

        # 1. Background & Canvas Setup (Strictly Untouched)
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        self.login_canvas = tk.Canvas(self.root, highlightthickness=0)
        self.login_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        def fill_background(event=None):
            # This line is the ONLY way to stop the "invalid command name .!canvas" error
            if self.current_page != "login": 
                return 

            nw, nh = self.root.winfo_width(), self.root.winfo_height()
            try:
                from PIL import ImageFilter, ImageEnhance
                img = Image.open(self.resource_path("SM 3 Bg.jpeg")).resize((nw, nh), Image.Resampling.LANCZOS)
                img = img.filter(ImageFilter.GaussianBlur(radius=12))
                img = ImageEnhance.Brightness(img).enhance(0.7)
                self.login_bg_img = ImageTk.PhotoImage(img)
                
                # This logic only runs if the login page is active
                self.login_canvas.delete("bg")
                self.login_canvas.create_image(0, 0, image=self.login_bg_img, anchor="nw", tags="bg")
                self.login_canvas.tag_lower("bg")
                
                self.login_canvas.coords(title_text, nw//2, nh//2 - 120)
                self.login_canvas.coords(entry_win, nw//2, nh//2 - 30)
                self.login_canvas.coords(btn_win, nw//2, nh//2 + 45)
                # Ensure forgot_text exists or remove this line if not used
                if 'forgot_text' in locals():
                    self.login_canvas.coords(forgot_text, nw//2, nh//2 + 110)
            except:
                # Extra check to prevent the second crash
                if hasattr(self, 'login_canvas') and self.login_canvas.winfo_exists():
                    self.login_canvas.configure(bg="black")

        # --- EDITED SECTION: Text Items (No Background Boxes) ---
        
        # USER LOGIN as Canvas Text (Zero background)
        title_text = self.login_canvas.create_text(
            w//2, h//2 - 120, 
            text="USER LOGIN", font=FONT_TITLE, fill=GOLD
        )

        # --- UNTOUCHED SECTION: Widgets ---
        
        pwd = tk.Entry(self.root, show="*", font=FONT_SUB, justify="center")
        entry_win = self.login_canvas.create_window(w//2, h//2 - 30, window=pwd, width=300)

       # --- LOGIN BUTTON WITH IMAGE (Button 11.jpeg) ---
        login_btn = self.image_button(
            self.root, self.resource_path("Button 11.jpeg"),
            lambda: self.dashboard() if pwd.get() == "12345"
            else messagebox.showerror("Error", "Invalid Password")
        )
        
        # This removes the black container edges and matches the background
        login_btn.config(bd=0, highlightthickness=0, bg="#050505", activebackground="#050505")
        
        # Re-using your original placement variable
        btn_win = self.login_canvas.create_window(w//2, h//2 + 45, window=login_btn)
        self.root.bind("<Configure>", fill_background)
        fill_background()

    # ============== DASHBOARD ==============
    def dashboard(self):
        self.current_page = "dashboard"
        self.clear()
        
        self.root.update()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        
        # 1. Canvas Setup (Ensure the canvas itself is black/theme colored)
        canvas = tk.Canvas(self.root, width=w, height=h, highlightthickness=0, bg="#050505")
        canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # --- LOCAL RESIZE FUNCTION ---
        def fill_dashboard_bg(event=None):
            if self.current_page != "dashboard": return
            
            nw, nh = self.root.winfo_width(), self.root.winfo_height()
            try:
                # Update Background
                img = Image.open(self.resource_path("SM 3 Bg.jpeg")).resize((nw, nh), Image.Resampling.LANCZOS)
                img = img.filter(ImageFilter.GaussianBlur(6))
                self.bg_img = ImageTk.PhotoImage(img)
                canvas.delete("bg_tag")
                canvas.create_image(0, 0, image=self.bg_img, anchor="nw", tags="bg_tag")
                canvas.tag_lower("bg_tag")
                
                # Move everything to new center coordinates
                canvas.coords(logo_item, nw//2, nh//2 - 120)
                canvas.coords(tagline_item, nw//2, nh//2 + 80)
                canvas.coords(btn_item, nw//2, nh//2 + 200)
            except:
                pass

        # --- LOGO (The absolute fix for the black box) ---
        try:
            # Force RGBA for transparency
            raw_logo = Image.open(self.resource_path("logo new.png")).convert("RGBA") 
            # Use LANCZOS to fix the 'D' and 'T'
            logo_resized = raw_logo.resize((500, 500), Image.Resampling.LANCZOS)
            self.logo_img = ImageTk.PhotoImage(logo_resized)
            
            # IMPORTANT: We use canvas.create_image (NOT create_window)
            # This allows the canvas to draw the logo pixels directly over the background
            canvas.delete("logo_tag")
            logo_item = canvas.create_image(w//2, h//2 - 120, image=self.logo_img, anchor="center", tags="logo_tag")
        except:
            logo_item = canvas.create_text(w//2, h//2 - 120, text="DAILY MART", fill="white")

        # --- TAGLINE ---
        tagline_item = canvas.create_text(w//2, h//2 + 80, 
                           text="Smart insights • Better decisions • Business growth",
                           font=("Times New Roman", 22, "bold"), fill="white")

        # --- EXPLORE BUTTON ---
        # The button is the ONLY thing allowed to be a 'window'
        explore_btn = self.image_button(self.root, self.resource_path("Button 8.jpeg"), self.feature_page)
        # These configs remove the black "sliver" edges around the capsule
        explore_btn.config(bd=0, highlightthickness=0, bg="#050505", activebackground="#050505")
        btn_item = canvas.create_window(w//2, h//2 + 200, window=explore_btn, anchor="center")

        self.root.bind("<Configure>", fill_dashboard_bg)
        fill_dashboard_bg()

    # ============== FEATURE PAGE ==============
    def feature_page(self):
        self.current_page = "feature"
        self.clear()

        # 1. Canvas Setup (Strictly Untouched)
        w, h = self.root.winfo_width(), self.root.winfo_height()
        # Initialize with explicit black background to prevent white flash
        canvas = tk.Canvas(self.root, width=w, height=h, highlightthickness=0, bg=BLACK)
        canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # --- LOCAL RESIZE LOGO (STOPS THE ERRORS) ---
        def fill_feature_bg(event=None):
            # THE KILL-SWITCH: Prevents "invalid command name" error
            if self.current_page != "feature": return 
            
            nw, nh = self.root.winfo_width(), self.root.winfo_height()
            try:
                # 2. BLURRED BACKGROUND IMAGE (Re-calculates on resize)
                img = Image.open(self.resource_path("SM 2 Bg.jpeg")).resize((nw, nh), Image.Resampling.LANCZOS)
                img = img.filter(ImageFilter.GaussianBlur(6)) 
                self.feature_bg = ImageTk.PhotoImage(img)
                
                canvas.delete("bg_tag")
                canvas.create_image(0, 0, image=self.feature_bg, anchor="nw", tags="bg_tag")
                canvas.tag_lower("bg_tag")
            except:
                canvas.configure(bg="#050505")

      
       # 3. BUTTON MAPPING (No Frame = No Black Box)
        button_configs = [
            ("Button 1.jpeg", self.load_dataset), ("Button 2.jpeg", self.data_preprocessing),
            ("Button 3.jpeg", self.eda_processing), ("Button 4.jpeg", self.model_training),
            ("Button 5.jpeg", self.results_analysis), ("Button 6.jpeg", self.outcome_analysis),
        ]

        # 4. PLACE INDIVIDUALLY ON CANVAS
        for i, (img_path, cmd) in enumerate(button_configs):
            r, c = divmod(i, 3)
            # Parent is self.root, but we anchor it to the canvas coordinate
            btn = self.image_button(self.root, self.resource_path(img_path), cmd)
            btn.config(bd=0, highlightthickness=0, bg="#050505", activebackground="#050505")
            
            # Calculate coordinates to mimic your grid exactly
            x_pos = (w // 2) + (c - 1) * 350
            y_pos = (h // 2) + (r - 0.5) * 150
            
            # This makes the button float on top of the blurred image
            canvas.create_window(x_pos, y_pos, window=btn)

        # 5. PREVIOUS BUTTON (Strictly Untouched)
        # 5. PREVIOUS BUTTON (Aligned exactly under Result & Analysis)
        prev_btn = self.image_button(self.root, self.resource_path("Button 7.jpeg"), self.dashboard)
        prev_btn.config(bd=0, highlightthickness=0, bg="#050505", activebackground="#050505")
        
        # X is w // 2 to match the middle column (Result & Analysis)
        # Y is set to + 270 to sit comfortably below the bottom row
        prev_win = canvas.create_window(w // 2, (h // 2) + 270, window=prev_btn, anchor="center")
        # Bind the resize and run it once
        self.root.bind("<Configure>", fill_feature_bg)
        fill_feature_bg()

    # ============== LOAD DATASET (Updated for Image Buttons) ==============
    def load_dataset(self):
        win = tk.Toplevel(self.root)
        win.title("Load Dataset")
        win.geometry("1000x600")
        win.configure(bg=BLACK)



        frame = tk.Frame(win, bg=BLACK)
        frame.pack(expand=True)

        # Use Image Buttons
        # Button 9: Upload Dataset (Assumed)
        btn_upload = self.image_button(frame, self.resource_path("Button 9.jpeg"), self.upload_dataset)
        btn_upload.grid(row=0, column=0, padx=30, pady=20)
        
        # Button 10: Recent Files (Assumed)
        btn_recent = self.image_button(frame, self.resource_path("Button 10.jpeg"), self.view_recent)
        btn_recent.grid(row=0, column=1, padx=30, pady=20)
        
        # Back Button (Button 7 is "Previous" usually)
        btn_back = self.image_button(frame, self.resource_path("Button 7.jpeg"), win.destroy)
        btn_back.grid(row=1, columnspan=2, pady=40)

    # ============== ENHANCED DATA PREPROCESSING MODULE ==============
    def data_preprocessing(self):
        if self.current_df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        # Create modal frame within main window instead of separate window
        self.clear()
        self.current_page = "preprocessing"
        
        # Create a frame that covers the entire window
        self.preprocessing_frame = tk.Frame(self.root, bg=BLACK)
        self.preprocessing_frame.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Title
        tk.Label(self.preprocessing_frame, text="Data Preprocessing - Structured Data Conversion",
                 font=FONT_TITLE, fg=GOLD, bg=BLACK).pack(pady=20)
        
        # Store original copy if not already stored
        if self.original_df is None:
            self.original_df = self.current_df.copy()
            self.preprocessing_steps = []
            self.brand_encoding_mapping = {}
        
        # Main container with left sidebar and right content
        main_container = tk.Frame(self.preprocessing_frame, bg=BLACK)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left Sidebar (30% width)
        sidebar = tk.Frame(main_container, bg=CARD, bd=2, relief="ridge")
        sidebar.pack(side='left', fill='y', padx=(0, 10))
        
        # Right Content Area (70% width)
        content_frame = tk.Frame(main_container, bg=BLACK, bd=2, relief="ridge")
        content_frame.pack(side='right', fill='both', expand=True)
        
        # Sidebar title
        tk.Label(sidebar, text="Preprocessing Menu", 
                font=FONT_SUB, fg=GOLD, bg=CARD, pady=10).pack(fill='x')
        
        # Sidebar separator
        tk.Frame(sidebar, height=2, bg=GOLD).pack(fill='x', pady=5)
        
        # Sidebar buttons
        sections = [
            ("📋 Data Overview", "overview"),
            ("🧹 Data Cleaning", "cleaning"),
            ("🔄 Data Transformation", "transformation"),
            ("🔤 Encoding Mappings", "mappings"),
            ("📜 Preprocessing History", "history")
        ]
        
        # Create a container for sidebar buttons
        button_container = tk.Frame(sidebar, bg=CARD)
        button_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        for i, (text, section_id) in enumerate(sections):
            btn = self.create_sidebar_button(button_container, text, section_id, i)
            btn.pack(fill='x', pady=5, ipady=8)
        
        # Initialize content area with overview
        self.current_preprocessing_section = "overview"
        self.show_preprocessing_section("overview", content_frame)
        
        # Bottom buttons (outside the sidebar/content area)
        button_frame = tk.Frame(self.preprocessing_frame, bg=BLACK)
        button_frame.pack(pady=10, fill='x', padx=20) # Add padding to container
        
        # Button removed from footer as per user request
        
        self.premium_button(button_frame, "View Processed Data", 
                           lambda: self.show_dataset(self.current_file), 
                           width=18, height=1).pack(side='left', padx=10, expand=True, fill='x')
        
        self.premium_button(button_frame, "Save Processed Data", 
                           self.save_processed_data, 
                           width=18, height=1).pack(side='left', padx=10, expand=True, fill='x')
        
        self.premium_button(button_frame, "Reset to Original", 
                           self.reset_to_original, 
                           width=18, height=1).pack(side='left', padx=10, expand=True, fill='x')
        
        self.premium_button(button_frame, "Previous", 
                           lambda: self.feature_page(), 
                           width=18, height=1).pack(side='left', padx=10, expand=True, fill='x')

    def create_sidebar_button(self, parent, text, section_id, index):
        """Create a sidebar button with hover effect"""
        btn_frame = tk.Frame(parent, bg=CARD, relief="flat")
        
        # Store section_id in button widget
        btn_frame.section_id = section_id
        
        # Button label
        label = tk.Label(btn_frame, text=text, font=FONT_SMALL, 
                        fg="white" if section_id != "overview" else GOLD, 
                        bg=CARD, anchor='w', padx=10)
        label.pack(fill='x', pady=5)
        
        # Hover effects
        def on_enter(e):
            if section_id != self.current_preprocessing_section:
                btn_frame.configure(bg="#1a1a1a")
                label.configure(bg="#1a1a1a")
        
        def on_leave(e):
            if section_id != self.current_preprocessing_section:
                btn_frame.configure(bg=CARD)
                label.configure(bg=CARD)
        
        def on_click(e):
            # Update all button colors
            for child in parent.winfo_children():
                if hasattr(child, 'section_id'):
                    child_label = child.winfo_children()[0]
                    if child.section_id == section_id:
                        child.configure(bg=GOLD)
                        child_label.configure(bg=GOLD, fg="black", font=("Times New Roman", 10, "bold"))
                    else:
                        child.configure(bg=CARD)
                        child_label.configure(bg=CARD, fg="white", font=FONT_SMALL)
            
            # Clear content frame and show new section
            for widget in parent.master.master.winfo_children()[1:]:
                widget.destroy()
            
            content_frame = tk.Frame(parent.master.master, bg=BLACK, bd=2, relief="ridge")
            content_frame.pack(side='right', fill='both', expand=True)
            
            self.current_preprocessing_section = section_id
            self.show_preprocessing_section(section_id, content_frame)
        
        btn_frame.bind("<Enter>", on_enter)
        btn_frame.bind("<Leave>", on_leave)
        btn_frame.bind("<Button-1>", on_click)
        label.bind("<Enter>", on_enter)
        label.bind("<Leave>", on_leave)
        label.bind("<Button-1>", on_click)
        
        return btn_frame

    def show_preprocessing_section(self, section_id, content_frame):
        """Display the selected section in the content frame"""
        # Clear content frame
        for widget in content_frame.winfo_children():
            widget.destroy()
        
        if section_id == "overview":
            self.show_data_overview(content_frame)
        elif section_id == "cleaning":
            self.show_data_cleaning(content_frame)
        elif section_id == "transformation":
            self.show_data_transformation(content_frame)
        elif section_id == "mappings":
            self.show_encoding_mappings(content_frame)
        elif section_id == "history":
            self.show_preprocessing_history(content_frame)

    def show_data_overview(self, parent):
        """Display data overview in the content area"""
        # Title
        tk.Label(parent, text="📋 Data Overview", 
                font=FONT_SUB, fg=GOLD, bg=BLACK).pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg=BLACK, highlightthickness=0)
        # Styled scrollbar (Standard tk scrollbar with dark colors - Note: Windows support varies)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, style="Dark.Vertical.TScrollbar")
        scrollable_frame = tk.Frame(canvas, bg=BLACK)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Data info
        info_frame = tk.Frame(scrollable_frame, bg="#1a1a1a", bd=1, relief="sunken")
        info_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        info_text = tk.Text(info_frame, height=20, width=80, bg="#1a1a1a", fg="white", 
                           font=FONT_CODE, wrap=tk.WORD)
        info_text.pack(padx=5, pady=5, fill='both', expand=True)
        
        info_text.tag_config("green", foreground="#20c997") # Teal/Green for data details
        info_text.tag_config("white", foreground="white")
        
        # Header Info
        info_text.insert('end', f"Dataset Shape: {self.current_df.shape}\n", "white")
        info_text.insert('end', f"Total Records: {len(self.current_df)}\n\n", "white")
        
        info_text.insert('end', "Column Information:\n", "white")
        
        for col in self.current_df.columns:
            dtype = self.current_df[col].dtype
            null_count = self.current_df[col].isnull().sum()
            unique_count = self.current_df[col].nunique()
            
            # Format: ColName: dtype | Null: x Unique: y (Green)
            info_text.insert('end', f"{col}: {dtype} | Null: {null_count} | Unique: {unique_count}\n", "green")
        
        info_text.config(state='disabled')

    def show_data_cleaning(self, parent):
        """Display data cleaning options in the content area"""
        # Title
        tk.Label(parent, text="🧹 Data Cleaning", 
                font=FONT_SUB, fg=GOLD, bg=BLACK).pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg=BLACK, highlightthickness=0)
        # Styled scrollbar (Standard tk scrollbar with dark colors - Note: Windows support varies)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, style="Dark.Vertical.TScrollbar")
        scrollable_frame = tk.Frame(canvas, bg=BLACK)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Missing Values Handling
        cleaning_frame = tk.Frame(scrollable_frame, bg=BLACK)
        cleaning_frame.pack(fill='both', expand=True, padx=10, pady=10)
        cleaning_frame.grid_columnconfigure(0, weight=1)
        cleaning_frame.grid_columnconfigure(1, weight=1)
        
        tk.Label(cleaning_frame, text="Handle Missing Values:", 
                font=FONT_SMALL, fg=GOLD, bg=BLACK).grid(row=0, column=0, sticky='w', pady=5)
        
        missing_method = tk.StringVar(value="mean")
        methods = [("Mean (Numerical)", "mean"), ("Median (Numerical)", "median"), 
                  ("Mode (Categorical)", "mode"), ("Drop Rows", "drop"), 
                  ("Drop Columns", "drop_cols"), ("Forward Fill", "ffill"), 
                  ("Backward Fill", "bfill")]
        
        for i, (text, value) in enumerate(methods):
            rb = tk.Radiobutton(cleaning_frame, text=text, variable=missing_method, 
                               value=value, bg=BLACK, fg="white", selectcolor=GOLD,
                               font=FONT_SMALL)
            rb.grid(row=i+1, column=0, sticky='w', padx=20, pady=2)
        
        def handle_missing_values():
            method = missing_method.get()
            # Architectural Fix: EDA/Cleaning must use UNENCODED data
            base_df = self.cleaned_df if self.cleaned_df is not None else self.original_df
            temp_df = base_df.copy()
            
            # AUTOMATIC: Correct time NaNs randomly before proceeding
            self.apply_random_time_fill_to_df(temp_df)
            
            cols_with_missing = temp_df.columns[temp_df.isnull().any()].tolist()
            
            if not cols_with_missing:
                # Update current_df even if no "missing" values found, because time fix might have happened
                self.current_df = temp_df
                self.show_message("Info", "No other missing values found! (Time corrected if needed)")
                return
            
            if method in ['mean', 'median']:
                # Apply only to numeric columns
                numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
                numeric_cols_with_missing = [col for col in cols_with_missing if col in numeric_cols]
                
                if numeric_cols_with_missing:
                    if method == 'mean':
                        temp_df[numeric_cols_with_missing] = temp_df[numeric_cols_with_missing].fillna(
                            temp_df[numeric_cols_with_missing].mean())
                        self.preprocessing_steps.append(f"Filled missing values with mean in columns: {numeric_cols_with_missing}")
                    else:
                        temp_df[numeric_cols_with_missing] = temp_df[numeric_cols_with_missing].fillna(
                            temp_df[numeric_cols_with_missing].median())
                        self.preprocessing_steps.append(f"Filled missing values with median in columns: {numeric_cols_with_missing}")
            elif method == 'mode':
                # For categorical columns - convert to title case first
                categorical_cols = temp_df.select_dtypes(exclude=[np.number]).columns
                cat_cols_with_missing = [col for col in cols_with_missing if col in categorical_cols]
                
                if cat_cols_with_missing:
                    for col in cat_cols_with_missing:
                        # Convert to string and title case
                        temp_df[col] = temp_df[col].astype(str).str.title()
                        # Preserve case sensitivity by using the actual mode value
                        mode_val = temp_df[col].mode()
                        if not mode_val.empty:
                            temp_df[col].fillna(mode_val[0], inplace=True)
                    self.preprocessing_steps.append(f"Filled missing values with mode in columns: {cat_cols_with_missing}")
            elif method == 'drop':
                initial_shape = temp_df.shape
                temp_df.dropna(inplace=True)
                final_shape = temp_df.shape
                rows_dropped = initial_shape[0] - final_shape[0]
                self.preprocessing_steps.append(f"Dropped {rows_dropped} rows with missing values")
            elif method == 'drop_cols':
                self.create_input_dialog("Drop Columns", 
                                       "Enter column names to drop (comma separated):",
                                       self.drop_columns_callback)
                return
            
            if method in ['ffill', 'bfill']:
                if method == 'ffill':
                    temp_df.fillna(method='ffill', inplace=True)
                    self.preprocessing_steps.append("Applied forward fill for missing values")
                else:
                    temp_df.fillna(method='bfill', inplace=True)
                    self.preprocessing_steps.append("Applied backward fill for missing values")
            
            # Update state
            self.current_df = temp_df
            self.cleaned_df = temp_df.copy()
            self.encoded_df = None # Invalidate encoded state
            
            # Apply brand name title case after missing value handling
            self.apply_brand_title_case()
            self.show_message("Success", "Missing values handled successfully!")
        

        self.premium_button(cleaning_frame, "Handle Missing Values", handle_missing_values, width=25, height=1).grid(row=len(methods)+1, column=0, pady=20, sticky='w', padx=20)
        
        # Outlier Detection and Treatment
        tk.Label(cleaning_frame, text="Outlier Treatment:", 
                font=FONT_SMALL, fg=GOLD, bg=BLACK).grid(row=0, column=1, sticky='w', padx=50, pady=5)
        
        def detect_outliers():
            numeric_cols = self.current_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                self.show_message("Info", "No numeric columns for outlier detection!")
                return
            
            outlier_info = "Outlier Detection Results:\n"
            outlier_info += "-" * 40 + "\n"
            
            for col in numeric_cols:
                Q1 = self.current_df[col].quantile(0.25)
                Q3 = self.current_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.current_df[(self.current_df[col] < lower_bound) | (self.current_df[col] > upper_bound)]
                outlier_count = len(outliers)
                
                outlier_info += f"{col}: {outlier_count} outliers detected\n"
            
            # Create custom confirmation dialog
            self.create_confirm_dialog("Outlier Detection", outlier_info, 
                                      "Do you want to remove outliers using IQR method?",
                                      self.remove_outliers_callback)
        
        self.premium_button(cleaning_frame, "Detect & Remove Outliers", detect_outliers, width=25, height=1).grid(row=1, column=1, padx=50, pady=10, sticky='w')
        
        # Brand Name Title Case Conversion Button
        tk.Label(cleaning_frame, text="Brand Name Formatting:", 
                font=FONT_SMALL, fg=GOLD, bg=BLACK).grid(row=2, column=1, sticky='w', padx=50, pady=5)
        
        def convert_brand_to_title_case():
            # Architectural Fix: Use UNENCODED base
            base_df = self.cleaned_df if self.cleaned_df is not None else self.original_df
            temp_df = base_df.copy()
            brand_cols = [col for col in temp_df.columns if 'brand' in col.lower()]
            
            if not brand_cols:
                self.show_message("Info", "No brand columns found!")
                return
            
            changes_made = False
            for col in brand_cols:
                # Store original values to compare
                original_sample = temp_df[col].head(10).tolist()
                
                # Convert to title case
                temp_df[col] = temp_df[col].astype(str).str.title()
                
                # Check if changes were made
                new_sample = temp_df[col].head(10).tolist()
                if original_sample != new_sample:
                    changes_made = True
            
            if changes_made:
                self.preprocessing_steps.append(f"Converted brand names to Title Case in columns: {brand_cols}")
                # Update State
                self.current_df = temp_df
                self.cleaned_df = temp_df.copy()
                self.encoded_df = None # Invalidate encoded
                self.show_message("Success", f"Brand names converted to Title Case!")
            else:
                self.show_message("Info", "Brand names are already in Title Case!")
        
        self.premium_button(cleaning_frame, "Convert Brand to Title Case", convert_brand_to_title_case, width=25, height=1).grid(row=3, column=1, padx=50, pady=10, sticky='w')

    def show_data_transformation(self, parent):
        """Display data transformation options in the content area"""
        # Title
        tk.Label(parent, text="🔄 Data Transformation", 
                font=FONT_SUB, fg=GOLD, bg=BLACK).pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg=BLACK, highlightthickness=0)
        # Styled scrollbar (Standard tk scrollbar with dark colors - Note: Windows support varies)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, style="Dark.Vertical.TScrollbar")
        scrollable_frame = tk.Frame(canvas, bg=BLACK)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Transformation options
        transform_frame = tk.Frame(scrollable_frame, bg=BLACK)
        transform_frame.pack(fill='both', expand=True, padx=10, pady=10)
        transform_frame.grid_columnconfigure(0, weight=1)
        transform_frame.grid_columnconfigure(1, weight=1)
        
        # Encoding categorical variables - INCLUDING BRAND NAMES
        tk.Label(transform_frame, text="Smart Label Encoding (<100 unique):", 
                font=FONT_SMALL, fg=GOLD, bg=BLACK).grid(row=0, column=0, sticky='w', pady=10)
        
        def encode_categorical():
            # Architectural Fix: EDA/Cleaning must use UNENCODED data
            # Transformation (Encoding) must derive from latest CLEANED data
            base_df = self.cleaned_df if self.cleaned_df is not None else self.original_df
            temp_df = base_df.copy()
            
            categorical_cols = temp_df.select_dtypes(exclude=[np.number]).columns
            
            if len(categorical_cols) == 0:
                self.show_message("Info", "No categorical columns found!")
                return
            
            # Check for specific column types
            brand_cols = [col for col in categorical_cols if 'brand' in col.lower()]
            product_cols = [col for col in categorical_cols if 'product' in col.lower() or 'item' in col.lower()]
            payment_cols = [col for col in categorical_cols if 'payment' in col.lower()]
            stock_cols = [col for col in categorical_cols if 'stock' in col.lower()]

            # Exclude specific fields
            excluded_fields = ['invoice id', 'date', 'time', 'product id', 'mobile number', 'product code']
            exclusions_normalized = [f.replace(' ', '').lower() for f in excluded_fields]
            
            # Create mapping dictionary to preserve original values
            encoding_info = "Encoding Applied:\n"
            encoding_info += "-" * 40 + "\n"
            
            le = LabelEncoder()
            encoded_cols = []
            
            # Encode ALL categorical columns found
            for col in categorical_cols:
                # Check for exclusion
                col_norm = col.lower().replace('_', ' ').replace(' ', '')
                if any(ex in col_norm for ex in exclusions_normalized) or col.lower() in excluded_fields:
                     continue

                # Store original values for key columns
                if col in brand_cols:
                    self.original_brand_names[col] = self.current_df[col].copy()
                elif col in product_cols:
                    self.original_product_names[col] = self.current_df[col].copy()
                elif col in payment_cols:
                    self.original_payment_methods[col] = self.current_df[col].copy()
                elif col in stock_cols:
                    self.original_stock_status[col] = self.current_df[col].copy()
                
                original_values = self.current_df[col].astype(str).unique()
                encoded_values = le.fit_transform(self.current_df[col].astype(str))
                
                # Create mapping dictionary
                mapping = dict(zip(original_values, encoded_values))
                self.brand_encoding_mapping[col] = mapping  # Store mapping
                
                for orig, enc in mapping.items():
                    encoding_info += f"  '{orig}' → {enc}\n"
                
                temp_df[col] = encoded_values
                encoded_cols.append(col)
                
                self.preprocessing_steps.append(f"Encoded categorical column: {col}")
            
            if encoded_cols:
                # Update State
                self.current_df = temp_df
                self.encoded_df = temp_df.copy() # Store for Model Training
                self.show_message("Encoding Complete", encoding_info)
            else:
                self.show_message("Info", "No suitable categorical columns for encoding")
        
        self.premium_button(transform_frame, "Apply Label Encoding", encode_categorical, width=25, height=1).grid(row=1, column=0, pady=25, sticky='w')
        
        # Encode Specific Columns Only Button
        tk.Label(transform_frame, text="Force Encode All Categorical:", 
                font=FONT_SMALL, fg=GOLD, bg=BLACK).grid(row=0, column=1, sticky='w', padx=50, pady=10)
        
        def encode_specific_columns():
            # --- FIX: ALWAYS USE CURRENT STATE (Allows chaining) ---
            base_df = self.current_df if self.current_df is not None else self.original_df
            if base_df is None:
                self.show_message("Error", "No dataset loaded!")
                return

            temp_df = base_df.copy()
            
            # Select all object/category columns
            categorical_cols = temp_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Identify Candidates
            candidates = []
            skipped_reasons = []

            # Exclusion list
            excluded_fields = ['invoice id', 'date', 'time', 'product id', 'mobile number', 'product code']
            exclusions_normalized = [f.replace(' ', '').lower() for f in excluded_fields]
            
            for col in categorical_cols:
                col_norm = col.lower().replace('_', ' ').replace(' ', '')
                
                # Check for exclusion
                is_excluded = False
                if any(ex == col_norm for ex in exclusions_normalized):
                     is_excluded = True
                     skipped_reasons.append(f" - {col}: Excluded (Restricted Field)")
                elif any(ex in col.lower().replace('_', ' ') for ex in excluded_fields):
                     is_excluded = True
                     skipped_reasons.append(f" - {col}: Excluded (Contains Restricted Keyword)")
                
                if not is_excluded:
                    candidates.append(col)

            if not candidates:
                msg = "No suitable columns found for encoding.\n\nSkipped Columns:\n" + "\n".join(skipped_reasons)
                self.show_message("Info", msg)
                return
            
            encoding_info = "Encoder Report:\n"
            encoding_info += "-" * 40 + "\n"
            
            le = LabelEncoder()
            encoded_cols = []
            
            for col in candidates:
                # Skip if already numeric (double check)
                if pd.api.types.is_numeric_dtype(temp_df[col]):
                    continue
                    
                # Store original values for decoding later
                if 'brand' in col.lower():
                    self.original_brand_names[col] = temp_df[col].copy()
                elif 'product' in col.lower() or 'item' in col.lower():
                    self.original_product_names[col] = temp_df[col].copy()
                elif 'payment' in col.lower():
                    self.original_payment_methods[col] = temp_df[col].copy()
                elif 'stock' in col.lower():
                    self.original_stock_status[col] = temp_df[col].copy()
                
                # Encode
                try:
                    # Calculate unique count for info
                    unique_vals = temp_df[col].astype(str).unique()
                    count = len(unique_vals)
                    
                    # Apply Encoding
                    encoded_values = le.fit_transform(temp_df[col].astype(str))
                    
                    mapping = dict(zip(unique_vals, encoded_values))
                    self.brand_encoding_mapping[col] = mapping
                    
                    temp_df[col] = encoded_values
                    encoded_cols.append(col)
                    
                    encoding_info += f"✔ Encoded: {col} ({count} unique values)\n"
                    self.preprocessing_steps.append(f"Encoded '{col}'")
                    
                except Exception as e:
                    encoding_info += f"❌ Failed: {col} - {str(e)}\n"

            # Report Skipped (Excluded) columns
            if skipped_reasons:
                 encoding_info += "\nSKIPPED (Restricted):\n" + "\n".join(skipped_reasons[:5]) 
                 if len(skipped_reasons) > 5: encoding_info += f"\n...and {len(skipped_reasons)-5} more."

            if encoded_cols:
                # Update State
                self.current_df = temp_df
                self.encoded_df = temp_df.copy() # Store for Model Training
                self.show_message("Encoding Complete", encoding_info)
            else:
                self.show_message("Info", "No columns were successfully encoded.\n" + encoding_info)

        
        self.premium_button(transform_frame, "Encode All Categorical", encode_specific_columns, width=25, height=1).grid(row=1, column=1, padx=50, pady=25, sticky='w')
        
        # Normalization/Standardization
        tk.Label(transform_frame, text="Normalize Numerical Data:", 
                font=FONT_SMALL, fg=GOLD, bg=BLACK).grid(row=2, column=0, sticky='w', pady=10)
        
        def normalize_data():
            # Architectural Fix: Scaling should cumulative after encoding
            base_df = self.encoded_df if self.encoded_df is not None else (self.cleaned_df if self.cleaned_df is not None else self.original_df)
            temp_df = base_df.copy()
            numeric_cols = list(temp_df.select_dtypes(include=[np.number]).columns)
            
            # EXCLUDE TARGET VARIABLE FROM STANDARDIZATION
            # We assume 'TotalSale' is the target, or ask user. 
            # For this requirement, we exclude 'TotalSale' and 'TotalAmount' specifically if present.
            # ALSO EXCLUDE ID/Protected fields
            cols_to_exclude = ['TotalSale', 'TotalAmount', 'total_sale', 'total_amount']
            excluded_fields = ['invoice id', 'date', 'time', 'product id', 'mobile number', 'product code']
            exclusions_normalized = [f.replace(' ', '').lower() for f in excluded_fields]
            
            final_numeric_cols = []
            for col in numeric_cols:
                 if col in cols_to_exclude:
                     continue
                 
                 # Check against excluded fields
                 col_norm = col.lower().replace('_', ' ').replace(' ', '')
                 if any(ex in col_norm for ex in exclusions_normalized) or col.lower() in excluded_fields:
                     continue
                 
                 final_numeric_cols.append(col)
            
            numeric_cols = final_numeric_cols

            if len(numeric_cols) > 0:
                # Standardize
                scaler = StandardScaler()
                temp_df[numeric_cols] = scaler.fit_transform(temp_df[numeric_cols])
                
                # Update State
                self.current_df = temp_df
                self.encoded_df = temp_df.copy()
                
                self.preprocessing_steps.append(f"Standardized numerical columns (target excluded): {list(numeric_cols)}")
                self.show_message("Success", "Data standardized (Target variable preserved)!")
            else:
                self.show_message("Info", "No numeric columns for normalization (Target is excluded).")
        
        self.premium_button(transform_frame, "Apply Standardization", normalize_data, width=25, height=1).grid(row=3, column=0, pady=25, sticky='w')

        # NEW: Calculate Total Sales Section
        tk.Label(transform_frame, text="Calculate Metrics:", 
                font=FONT_SMALL, fg=GOLD, bg=BLACK).grid(row=2, column=1, sticky='w', padx=50, pady=10)
        
        def calculate_total_sales():
            if self.current_df is None:
                self.show_message("Error", "No dataset loaded!")
                return
            
            df = self.current_df
            cols = [c.lower() for c in df.columns]
            
            # Smart detection
            price_col = next((c for c in df.columns if any(x in c.lower() for x in ['price', 'amount', 'cost', 'rate'])), None)
            qty_col = next((c for c in df.columns if any(x in c.lower() for x in ['qty', 'quantity', 'count', 'vol'])), None)
            
            if not price_col or not qty_col:
                self.show_message("Error", f"Could not automatically detect Price/Quantity columns.\nFound: Price='{price_col}', Qty='{qty_col}'")
                return

            try:
                # Ensure numeric
                p_numeric = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
                q_numeric = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
                
                # Calculation
                df['Total Sales'] = p_numeric * q_numeric
                
                self.current_df = df
                self.encoded_df = df.copy() # Sync
                self.preprocessing_steps.append(f"Calculated 'Total Sales' = {price_col} * {qty_col}")
                self.show_message("Success", f"Created 'Total Sales' column!\nUsed: {price_col} * {qty_col}")
                
            except Exception as e:
                self.show_message("Error", f"Calculation failed: {str(e)}")

        self.premium_button(transform_frame, "Calculate Total Sale", calculate_total_sales, width=25, height=1).grid(row=3, column=1, padx=50, pady=25, sticky='w')

    def show_encoding_mappings(self, parent):
        """Display encoding mappings in the content area"""
        # Title
        tk.Label(parent, text="🔤 Encoding Mappings", 
                font=FONT_SUB, fg=GOLD, bg=BLACK).pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg=BLACK, highlightthickness=0)
        # Styled scrollbar (Standard tk scrollbar with dark colors - Note: Windows support varies)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, style="Dark.Vertical.TScrollbar")
        scrollable_frame = tk.Frame(canvas, bg=BLACK)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Encoding mappings text
        mapping_text = tk.Text(scrollable_frame, height=20, width=80, bg="#1a1a1a", fg="white", 
                              font=FONT_CODE, wrap=tk.WORD)
        mapping_text.pack(padx=10, pady=10, fill='both', expand=True)
        
        if self.brand_encoding_mapping:
            mapping_str = "Encoding Mappings:\n"
            mapping_str += "-" * 50 + "\n"
            for col, mapping in self.brand_encoding_mapping.items():
                mapping_str += f"\n{col}:\n"
                for brand_name, encoded_value in mapping.items():
                    mapping_str += f"  '{brand_name}' → {encoded_value}\n"
        else:
            mapping_str = "No encoding mappings yet. Apply encoding to see mappings."
        
        mapping_text.insert('1.0', mapping_str)
        mapping_text.config(state='disabled')

    def show_preprocessing_history(self, parent):
        """Display preprocessing history in the content area"""
        # Title
        tk.Label(parent, text="📜 Preprocessing History", 
                font=FONT_SUB, fg=GOLD, bg=BLACK).pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg=BLACK, highlightthickness=0)
        # Styled scrollbar (Standard tk scrollbar with dark colors - Note: Windows support varies)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, style="Dark.Vertical.TScrollbar")
        scrollable_frame = tk.Frame(canvas, bg=BLACK)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Preprocessing history text
        history_text = tk.Text(scrollable_frame, height=20, width=80, bg="#1a1a1a", fg="white", 
                              font=FONT_CODE, wrap=tk.WORD)
        history_text.pack(padx=10, pady=10, fill='both', expand=True)
        
        if self.preprocessing_steps:
            history_str = "Preprocessing Steps Applied:\n"
            history_str += "-" * 50 + "\n"
            for i, step in enumerate(self.preprocessing_steps, 1):
                history_str += f"{i}. {step}\n"
        else:
            history_str = "No preprocessing steps applied yet."
        
        history_text.insert('1.0', history_str)
        history_text.config(state='disabled')

    def apply_brand_title_case(self):
        """Convert all brand columns to title case"""
        brand_cols = [col for col in self.current_df.columns if 'brand' in col.lower()]
        for col in brand_cols:
            # Convert to string and apply title case
            self.current_df[col] = self.current_df[col].astype(str).str.title()

    def apply_random_time_fill_to_df(self, df):
        """Automatically detect time columns and fill NaNs with random HH:MM"""
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        changes = False
        for col in time_cols:
            mask = df[col].isna()
            if mask.any():
                df.loc[mask, col] = [f"{random.randint(0, 23):02}:{random.randint(0, 59):02}" for _ in range(mask.sum())]
                changes = True
        return changes

    def apply_random_time_fill(self):
        """Helper for self.current_df"""
        if self.current_df is not None:
            if self.apply_random_time_fill_to_df(self.current_df):
                 self.preprocessing_steps.append("Automatically corrected missing time values with random imputation")

    def create_input_dialog(self, title, prompt, callback):
        """Create custom input dialog within preprocessing frame"""
        overlay = tk.Frame(self.preprocessing_frame, bg="black", highlightthickness=0)
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        overlay.lift()
        
        dialog = tk.Frame(overlay, bg=BLACK, bd=3, relief="ridge")
        dialog.place(relx=0.5, rely=0.5, anchor="center", width=500, height=300)
        
        tk.Label(dialog, text=title, font=FONT_SUB, fg=GOLD, bg=BLACK).pack(pady=10)
        tk.Label(dialog, text=prompt, font=FONT_SMALL, fg="white", bg=BLACK).pack(pady=5)
        
        entry = tk.Entry(dialog, font=FONT_SMALL, width=40)
        entry.pack(pady=10)
        entry.focus_set()
        
        def on_submit():
            value = entry.get()
            overlay.destroy()
            if callback:
                callback(value)
        
        def on_cancel():
            overlay.destroy()
        
        btn_frame = tk.Frame(dialog, bg=BLACK)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="OK", command=on_submit, bg=GOLD, fg=BLACK,
                 font=FONT_SMALL, width=10).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Cancel", command=on_cancel, bg=GOLD, fg=BLACK,
                 font=FONT_SMALL, width=10).pack(side='left', padx=5)
        
        dialog.bind("<Return>", lambda e: on_submit())
        dialog.bind("<Escape>", lambda e: on_cancel())

    def drop_columns_callback(self, cols_to_drop):
        if cols_to_drop:
            cols_list = [col.strip() for col in cols_to_drop.split(',')]
            cols_dropped = [col for col in cols_list if col in self.current_df.columns]
            if cols_dropped:
                self.current_df.drop(columns=cols_dropped, inplace=True)
                self.preprocessing_steps.append(f"Dropped columns: {cols_dropped}")
                # Apply brand name title case after dropping columns
                self.apply_brand_title_case()
                self.show_message("Success", f"Dropped columns: {cols_dropped}")

    def create_confirm_dialog(self, title, message, question, callback):
        """Create custom confirmation dialog within preprocessing frame"""
        overlay = tk.Frame(self.preprocessing_frame, bg="black", highlightthickness=0)
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        overlay.lift()
        
        dialog = tk.Frame(overlay, bg=BLACK, bd=3, relief="ridge")
        dialog.place(relx=0.5, rely=0.5, anchor="center", width=600, height=400)
        
        tk.Label(dialog, text=title, font=FONT_SUB, fg=GOLD, bg=BLACK).pack(pady=10)
        
        # Message text
        msg_text = tk.Text(dialog, height=8, width=60, bg="#1a1a1a", fg="white",
                          font=FONT_SMALL, wrap=tk.WORD)
        msg_text.pack(pady=5, padx=10)
        msg_text.insert('1.0', message)
        msg_text.config(state='disabled')
        
        tk.Label(dialog, text=question, font=FONT_SMALL, fg="yellow", bg=BLACK).pack(pady=5)
        
        def on_yes():
            overlay.destroy()
            if callback:
                callback()
        
        def on_no():
            overlay.destroy()
        
        btn_frame = tk.Frame(dialog, bg=BLACK)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Yes", command=on_yes, bg=GOLD, fg=BLACK,
                 font=FONT_SMALL, width=10).pack(side='left', padx=5)
        tk.Button(btn_frame, text="No", command=on_no, bg=GOLD, fg=BLACK,
                 font=FONT_SMALL, width=10).pack(side='left', padx=5)
        
        dialog.bind("<Escape>", lambda e: on_no())

    def remove_outliers_callback(self):
        # Architectural Fix: Use UNENCODED base
        base_df = self.cleaned_df if self.cleaned_df is not None else self.original_df
        temp_df = base_df.copy()
        
        numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
        initial_shape = temp_df.shape
        for col in numeric_cols:
            Q1 = temp_df[col].quantile(0.25)
            Q3 = temp_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            temp_df = temp_df[(temp_df[col] >= lower_bound) & 
                             (temp_df[col] <= upper_bound)]
        
        # Update Master State
        self.current_df = temp_df
        self.cleaned_df = temp_df.copy()
        self.encoded_df = None # Invalidate encoded
        
        final_shape = self.current_df.shape
        rows_removed = initial_shape[0] - final_shape[0]
        self.preprocessing_steps.append(f"Removed {rows_removed} rows with outliers")
        
        # Apply brand name title case after outlier removal
        self.apply_brand_title_case()
        
        self.show_message("Success", f"Removed {rows_removed} outlier rows!")

    def show_message(self, title, message):
        """Show a message dialog"""
        overlay = tk.Frame(self.preprocessing_frame, bg="black", highlightthickness=0)
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        overlay.lift()
        
        dialog = tk.Frame(overlay, bg=BLACK, bd=3, relief="ridge")
        dialog.place(relx=0.5, rely=0.5, anchor="center", width=450, height=300)
        
        tk.Label(dialog, text=title, font=FONT_SUB, fg=GOLD, bg=BLACK).pack(pady=10)
        
        # Message text
        msg_text = tk.Text(dialog, height=5, width=40, bg="#1a1a1a", fg="white",
                          font=FONT_SMALL, wrap=tk.WORD)
        msg_text.pack(pady=5, padx=10)
        msg_text.insert('1.0', message)
        msg_text.config(state='disabled')
        
        def close_dialog():
            overlay.destroy()
        
        btn_frame = tk.Frame(dialog, bg=BLACK)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="OK", command=close_dialog, bg=GOLD, fg=BLACK,
                 font=FONT_SMALL, width=10).pack()
        
        dialog.bind("<Return>", lambda e: close_dialog())
        dialog.bind("<Escape>", lambda e: close_dialog())

    def save_processed_data(self):
        if self.current_df is None:
            return
        
        # Determine the best parent window
        parent_win = self.root
        if self.dataset_window and tk.Toplevel.winfo_exists(self.dataset_window):
            parent_win = self.dataset_window
        elif hasattr(self, 'preprocessing_frame') and self.preprocessing_frame.winfo_exists():
            parent_win = self.preprocessing_frame
            
        # Use custom file dialog
        file_path = filedialog.asksaveasfilename(
            parent=parent_win,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.current_df.to_csv(file_path, index=False)
                else:
                    self.current_df.to_excel(file_path, index=False)
                self.show_message("Success", f"Data saved to {os.path.basename(file_path)}")
            except Exception as e:
                self.show_message("Error", f"Failed to save data: {e}")

    def reset_to_original(self):
        if self.original_df is not None:
            self.current_df = self.original_df.copy()
            self.cleaned_df = None # Reset cleaned data
            self.encoded_df = None # Reset encoded data
            self.preprocessing_steps = []
            self.brand_encoding_mapping = {}
            self.original_brand_names = {}
            self.original_product_names = {}
            self.original_payment_methods = {}
            self.original_stock_status = {}
            
            # CRITICAL: Clear model state on data reset + Clear Results Page
            self.clear_model_state()
            self.model_results = {} 
            
            self.show_message("Reset", "Data reset to original state! (Model cleared)")

    # ============== DATASET VIEW + CRUD ==============
    def upload_dataset(self):
        file = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("All files", "*.*")]
        )
        if file:
            self.recent_files.append(file)
            self.current_file = file
            
            # Try to detect delimiter for CSV
            if file.endswith(".csv"):
                try:
                    # Try reading with different delimiters
                    with open(file, 'r') as f:
                        sample = f.read(1024)
                    
                    # Check for common delimiters
                    if ';' in sample and ',' not in sample.split('\n')[0]:
                        self.current_df = pd.read_csv(file, delimiter=';')
                    elif '\t' in sample:
                        self.current_df = pd.read_csv(file, delimiter='\t')
                    else:
                        self.current_df = pd.read_csv(file)
                except:
                    # If all else fails, try with engine='python'
                    self.current_df = pd.read_csv(file, engine='python')
            elif file.endswith(".xlsx"):
                self.current_df = pd.read_excel(file)
            
            # Reset dataframes - Master Roles
            self.original_df = self.current_df.copy()
            self.cleaned_df = None 
            self.encoded_df = None
            self.preprocessing_steps = []
            self.brand_encoding_mapping = {}
            self.original_brand_names = {}
            self.original_product_names = {}
            self.original_payment_methods = {}
            self.original_stock_status = {}
            
            # Clean column names (remove whitespace, special characters)
            self.current_df.columns = self.current_df.columns.str.strip()
            self.current_df.columns = [col.replace(' ', '_').replace('.', '_') for col in self.current_df.columns]
            
            # AUTOMATIC: Correct time NaNs on upload
            self.apply_random_time_fill()
            
            # Store original values for all relevant columns
            for col in self.current_df.columns:
                col_lower = col.lower()
                if 'brand' in col_lower:
                    self.original_brand_names[col] = self.current_df[col].copy()
                elif 'product' in col_lower or 'item' in col_lower:
                    self.original_product_names[col] = self.current_df[col].copy()
                elif 'payment' in col_lower:
                    self.original_payment_methods[col] = self.current_df[col].copy()
                elif 'stock' in col_lower:
                    self.original_stock_status[col] = self.current_df[col].copy()
            
            self.show_dataset(file)

    def show_dataset(self, file):
        if self.current_df is None:
            return

        if self.dataset_window is not None and tk.Toplevel.winfo_exists(self.dataset_window):
            self.dataset_window.destroy()

        win = tk.Toplevel(self.root)
        self.dataset_window = win
        win.title("Dataset View")
        win.geometry("1100x600")
        try:
            win.state('zoomed')
        except:
            pass # Fallback for non-windows
        win.configure(bg=BLACK)

        # SMALL BACK BUTTON (Top-Left)
        tk.Button(win, text="⬅", command=win.destroy,
                 font=("Times New Roman", 10, "bold"), bg=GOLD, fg=BLACK,
                 relief="raised", bd=1).place(x=10, y=10)

        tk.Label(win, text=f"{os.path.basename(file)} (Processed)" if len(self.preprocessing_steps) > 0 else os.path.basename(file),
                 font=FONT_SUB, fg=GOLD, bg=BLACK).pack(pady=10)

        # Show preprocessing info if any
        if len(self.preprocessing_steps) > 0:
            info_label = tk.Label(win, 
                text=f"✓ {len(self.preprocessing_steps)} preprocessing steps applied",
                font=FONT_SMALL, fg=GREEN, bg=BLACK)
            info_label.pack()

        style = ttk.Style()
        # style.theme_use("default")
        style.configure("Treeview",
                        background=BLACK,
                        foreground="white",
                        fieldbackground=BLACK,
                        rowheight=28,
                        font=FONT_SMALL)
        style.configure("Treeview.Heading",
                        background=GOLD,
                        foreground=BLACK,
                        font=FONT_BTN)

        table_frame = tk.Frame(win, bg=BLACK)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        cols = list(self.current_df.columns)
        tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        self.dataset_tree = tree

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=150, anchor="center")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Insert data (limit to 1000 rows for performance)
        sample_df = self.current_df.head(1000)
        for _, row in sample_df.iterrows():
            tree.insert("", "end", values=list(row))

        # If more than 1000 rows, show message
        if len(self.current_df) > 1000:
            tk.Label(win, text=f"Showing first 1000 of {len(self.current_df)} rows",
                    font=FONT_SMALL, fg="yellow", bg=BLACK).pack()

        ops = tk.Frame(win, bg=BLACK)
        ops.pack(pady=10)

        self.premium_button(ops, "Add", self.add_record_popup).pack(side="left", padx=10)
        self.premium_button(ops, "Edit", self.edit_record_popup).pack(side="left", padx=10)
        self.premium_button(ops, "Delete", self.delete_record).pack(side="left", padx=10)
        self.premium_button(ops, "Save", self.save_processed_data).pack(side="left", padx=10)
        self.premium_button(ops, "Back", win.destroy).pack(side="left", padx=20)

    # ---------- CRUD HELPERS ----------
    def add_record_popup(self):
        if self.dataset_tree is None or self.current_df is None:
            return
        self.record_form_popup(mode="add")

    def edit_record_popup(self):
        if self.dataset_tree is None or self.current_df is None:
            return
        selected = self.dataset_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a row to edit")
            return
        self.record_form_popup(mode="edit", item_id=selected[0])

    def record_form_popup(self, mode="add", item_id=None):
        cols = list(self.current_df.columns)

        popup = tk.Toplevel(self.root)
        popup.title("Add Record" if mode == "add" else "Edit Record")
        popup.geometry("550x500")
        popup.configure(bg=BLACK)

        tk.Label(popup,
                 text="Add New Record" if mode == "add" else "Edit Record",
                 font=FONT_SUB, fg=GOLD, bg=BLACK).pack(pady=10)

        # SCROLLABLE CONTAINER
        container = tk.Frame(popup, bg=BLACK)
        container.pack(fill="both", expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(container, bg=BLACK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        form_frame = tk.Frame(canvas, bg=BLACK)
        
        form_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=form_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        entries = {}
        current_values = None
        if mode == "edit" and item_id is not None:
            current_values = self.dataset_tree.item(item_id, "values")

        for i, col in enumerate(cols):
            tk.Label(form_frame, text=col, font=FONT_SMALL,
                     fg="white", bg=BLACK, anchor="w", width=18).grid(
                row=i, column=0, pady=5, padx=10, sticky="w"
            )
            e = tk.Entry(form_frame, font=FONT_SMALL, width=30)
            e.grid(row=i, column=1, pady=5, padx=10)
            if current_values is not None:
                try:
                    e.insert(0, current_values[i])
                except:
                    pass
            entries[col] = e

        def save_record():
            values = [entries[c].get() for c in cols]
            if mode == "add":
                new_row = pd.DataFrame([values], columns=cols)
                self.current_df = pd.concat([self.current_df, new_row], ignore_index=True)
                self.dataset_tree.insert("", "end", values=values)
            else:
                if item_id is None:
                    return
                self.dataset_tree.item(item_id, values=values)
                index = self.dataset_tree.index(item_id)
                self.current_df.loc[index, :] = values
            
            # Unbind mousewheel to prevent errors after destroy
            canvas.unbind_all("<MouseWheel>")
            popup.destroy()

        # Save button outside scroll area
        btn_frame = tk.Frame(popup, bg=BLACK)
        btn_frame.pack(pady=10)
        self.premium_button(btn_frame, "Save", save_record).pack()

    def delete_record(self):
        if self.dataset_tree is None or self.current_df is None:
            return
        selected = self.dataset_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a row to delete")
            return
        item_id = selected[0]
        index = self.dataset_tree.index(item_id)
        if not messagebox.askyesno("Confirm", "Delete selected record?"):
            return
        self.dataset_tree.delete(item_id)
        self.current_df = self.current_df.drop(self.current_df.index[index]).reset_index(drop=True)

    # ============== RECENT DATASET WINDOW ==============
    def view_recent(self):
        win = tk.Toplevel(self.root)
        win.title("Recent Datasets")
        win.geometry("420x260")
        win.configure(bg=BLACK)

        tk.Label(
            win,
            text="Recently Viewed Datasets",
            font=FONT_SUB,
            fg=GOLD,
            bg=BLACK
        ).pack(pady=15)

        if not self.recent_files:
            tk.Label(
                win,
                text="No recent files",
                font=FONT_SMALL,
                fg="white",
                bg=BLACK
            ).pack(pady=10)
            return

        lb = tk.Listbox(
            win,
            font=FONT_SMALL,
            bg="#1a1a1a",
            fg="white",
            selectbackground=GOLD,
            selectforeground=BLACK,
            height=min(8, len(self.recent_files)),
            activestyle="none"
        )
        lb.pack(fill="both", expand=True, padx=20, pady=10)

        files_to_show = list(reversed(self.recent_files[-10:]))
        for f in files_to_show:
            lb.insert("end", os.path.basename(f))

        def on_open(event):
            sel = lb.curselection()
            if not sel:
                return
            idx = sel[0]
            file_path = files_to_show[idx]
        lb.bind("<Double-Button-1>", on_open)

    # ============ DECODING HELPER ============
    def get_decoded_column(self, column_name):
        """Get decoded values for a specific column using current dataframe."""
        if self.current_df is None or column_name not in self.current_df.columns:
            return None
        
        # Check if column is numeric (likely encoded)
        if self.current_df[column_name].dtype in ['int64', 'float64']:
            # Decode using stored mappings/originals
            if column_name in self.original_brand_names and len(self.current_df[column_name]) == len(self.original_brand_names[column_name]):
                return self.original_brand_names[column_name]
            if column_name in self.original_product_names and len(self.current_df[column_name]) == len(self.original_product_names[column_name]):
                return self.original_product_names[column_name]
            if column_name in self.brand_encoding_mapping:
                 # Reverse mapping
                 reverse_map = {v: k for k, v in self.brand_encoding_mapping[column_name].items()}
                 return self.current_df[column_name].map(reverse_map)
            return self.current_df[column_name]
        else:
            return self.current_df[column_name]

    # ============== EDA PROCESSING MODULE ==============
    def eda_processing(self):
        if self.current_df is None:
            messagebox.showwarning("Warning", "Please load and preprocess a dataset first!")
            return

        self.current_page = "eda_processing"
        self.clear()
        
        # STRICT: Use cleaned_df (unencoded) or original_df for EDA. NEVER use current_df which may be encoded.
        target_df = self.cleaned_df if self.cleaned_df is not None else self.original_df
        target_df = target_df.copy() # Avoid SettingWithCopy warning
        
        # --- FIX: REMOVE DUPLICATES FOR EDA ---
        # User requested not to see duplicate values in charts
        target_df.drop_duplicates(inplace=True)
        
        # --- FIX: DECODE CATEGORICAL VARIABLES ---
        # If the data is already encoded (e.g. from previous steps), we must decode it back for EDA visualization
        if self.brand_encoding_mapping:
            for col, mapping in self.brand_encoding_mapping.items():
                if col in target_df.columns:
                    # Create reverse mapping: {0: 'Adidas', 1: 'Nike'}
                    reverse_map = {v: k for k, v in mapping.items()}
                    # Apply mapping - this requires the column to be consistently typed (int/float usually)
                    # We use .map() which is efficient
                    try:
                        target_df[col] = target_df[col].map(reverse_map).fillna(target_df[col])
                    except:
                        pass # verification safety
        
        # --- FIX: STANDARDIZE CATEGORICAL STRINGS ---
        # To prevent "Cash" and "cash" appearing as separate slices in charts
        cat_cols = target_df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            try:
                # Convert to string, strip, title case
                target_df[col] = target_df[col].astype(str).str.strip().str.title()
                # Revert 'Nan', 'None', 'N/A' etc back to np.nan so charts ignore them
                target_df[col].replace(['Nan', 'None', 'Na', 'N/a', 'Null', ''], np.nan, inplace=True)
            except:
                pass
        
        # --- PROJECT SPECIFIC INSIGHTS ---

        
        # --- SCROLLBAR STYLE ---
        style = ttk.Style()
        style.theme_use('clam')
        # Attempt to make scrollbar dark and "visible" (Gold Trough/Dark Handle or vice versa)
        style.configure("EDA.Vertical.TScrollbar", 
                        gripcount=0,
                        background="#333", 
                        darkcolor="#333", 
                        lightcolor="#333",
                        troughcolor="black", 
                        bordercolor="#333", 
                        arrowcolor=GOLD)
        style.map("EDA.Vertical.TScrollbar", background=[('active', GOLD)])


        # --- UI SETUP ---
        # Main container
        self.eda_frame = tk.Frame(self.root, bg=BLACK)
        self.eda_frame.place(x=0, y=0, relwidth=1, relheight=1)
        
        # 1. HEADER & TOP BANNER
        header_frame = tk.Frame(self.eda_frame, bg=BLACK)
        header_frame.pack(side="top", fill="x", padx=20, pady=10)
        
        tk.Label(header_frame, text="Exploratory Data Analysis", font=FONT_TITLE, fg=GOLD, bg=BLACK).pack(side="left")
        self.premium_button(header_frame, "Exit Dashboard", self.feature_page, width=15).pack(side="right")
        
        # 2. METRICS BANNER
        metrics_frame = tk.Frame(self.eda_frame, bg="#111", bd=1, relief="solid", highlightbackground="#333")
        metrics_frame.pack(fill="x", padx=20, pady=10)
        
        n_rows, n_cols = target_df.shape
        n_missing = target_df.isnull().sum().sum()
        num_cols = len(target_df.select_dtypes(include=[np.number]).columns)
        cat_cols = len(target_df.select_dtypes(exclude=[np.number]).columns)
        
        metrics = [("Total Rows", f"{n_rows:,}"), ("Total Columns", f"{n_cols}"), ("Missing Values", f"{n_missing}"),
                   ("Numerical Features", f"{num_cols}"), ("Categorical Features", f"{cat_cols}")]
        
        for i, (label, val) in enumerate(metrics):
            m_cont = tk.Frame(metrics_frame, bg="#111")
            m_cont.pack(side="left", fill="x", expand=True, pady=10)
            tk.Label(m_cont, text=label, font=("Times New Roman", 9), fg="gray", bg="#111").pack()
            tk.Label(m_cont, text=val, font=("Times New Roman", 18, "bold"), fg=GOLD, bg="#111").pack()
            if i < len(metrics) - 1: tk.Frame(metrics_frame, width=1, height=40, bg="#333").pack(side="left", fill="y", pady=10)



        # 3. NAVIGATION TABS
        nav_frame = tk.Frame(self.eda_frame, bg=BLACK)
        nav_frame.pack(fill="x", padx=20, pady=(10, 0))
        
        self.active_tab = "business"
        self.tab_buttons = {}

        def switch_content(section):
            self.active_tab = section
            for key, btn in self.tab_buttons.items():
                if key == section: btn.config(bg="#1A1A1A", fg=GOLD, bd=1, relief="solid")
                else: btn.config(bg=BLACK, fg="gray", bd=0, relief="flat")

            for widget in content_area.winfo_children(): widget.destroy()

            if section == "business": show_business_insights(content_area)
            elif section == "stats": show_statistics(content_area)
            elif section == "corr": show_correlation(content_area)
            elif section == "dist": show_distribution(content_area)

        tabs = [("Business Insights", "business"), ("Statistical Analysis", "stats"), ("Correlation Analysis", "corr"), ("Distribution Analysis", "dist")]
        for text, key in tabs:
            btn = tk.Button(nav_frame, text=text, font=("Times New Roman", 11, "bold"), bg=BLACK, fg="gray", 
                            activebackground="#1A1A1A", activeforeground=GOLD, cursor="hand2", padx=20, pady=8,
                            command=lambda k=key: switch_content(k))
            btn.pack(side="left", padx=(0, 5))
            
            # Hover effect for tabs
            def on_enter_tab(e, b=btn):
                if b['bg'] != "#1A1A1A": # If not active
                     b.config(bg="#222")
            def on_leave_tab(e, b=btn):
                if b['bg'] != "#1A1A1A": # If not active
                     b.config(bg=BLACK)
            
            btn.bind("<Enter>", on_enter_tab)
            btn.bind("<Leave>", on_leave_tab)
            
            self.tab_buttons[key] = btn

        # 4. CONTENT AREA
        content_area = tk.Frame(self.eda_frame, bg=BLACK)
        content_area.pack(fill="both", expand=True, padx=20, pady=10)

        # --- SHARED SCROLLABLE CONTAINER HELPER ---
        def create_scrollable_container(parent):
            canvas = tk.Canvas(parent, bg=BLACK, highlightthickness=0)
            scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, style="EDA.Vertical.TScrollbar")
            scrollable_frame = tk.Frame(canvas, bg=BLACK)

            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=parent.winfo_width()-30)
            
            def on_canvas_configure(event): canvas.itemconfig(canvas.find_withtag("all")[0], width=event.width)
            canvas.bind("<Configure>", on_canvas_configure)
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            def _on_mousewheel(event): canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            
            return scrollable_frame

        # --- SECTIONS ---
        def show_business_insights(parent):
            scrollable_frame = create_scrollable_container(parent)

            # Helpers
            def create_section(title, plot_func, insight_gen_func):
                section_frame = tk.Frame(scrollable_frame, bg=BLACK, pady=20)
                section_frame.pack(fill="x", padx=10)
                
                # Title
                tk.Label(section_frame, text=title, font=("Times New Roman", 16, "bold"), fg=GOLD, bg=BLACK).pack(anchor="w", pady=(0, 10))
                
                # Split View
                split = tk.Frame(section_frame, bg=BLACK)
                split.pack(fill="x", expand=True)
                
                # Left: Chart
                chart_cont = tk.Frame(split, bg="#0f0f0f", bd=1, relief="solid", highlightbackground="#333", height=500)
                chart_cont.pack(side="left", fill="both", expand=True, padx=(0, 10))
                
                # Right: Insight Container
                insight_cont = tk.Frame(split, bg="#0f0f0f", bd=1, relief="solid", highlightbackground="#333", width=350)
                insight_cont.pack(side="right", fill="y")
                insight_cont.pack_propagate(False)
                
                # Header: Insights
                tk.Label(insight_cont, text="INSIGHTS", font=("Times New Roman", 12, "bold", "underline"), fg=GOLD, bg="#0f0f0f", anchor="w").pack(fill="x", padx=15, pady=(15, 5))
                
                # Render content
                try:
                    stats = plot_func(chart_cont)
                    insight_gen_func(insight_cont, stats)
                except Exception as e:
                    tk.Label(chart_cont, text=f"Error: {str(e)}", fg="red", bg="#0f0f0f").pack()
                
                # Separator
                tk.Frame(scrollable_frame, height=1, bg="#333").pack(fill="x", pady=20)

            # Data Prep (Refined heuristic to favor Names over IDs)
            def get_col(keywords, exclude=None):
                if exclude is None: exclude = []
                
                # First Pass: Direct matches for descriptive keywords
                for kw in keywords:
                    for c in target_df.columns:
                        c_low = c.lower()
                        # If kw is exactly the column name (case insensitive)
                        if kw == c_low: return c
                
                # Second Pass: Substring matches with exclusion
                for kw in keywords:
                    for c in target_df.columns:
                        c_low = c.lower()
                        if kw in c_low:
                            # Skip if column contains any exclusion keyword (e.g., 'id', 'code')
                            if not any(ex.lower() in c_low for ex in exclude):
                                return c
                
                # Third Pass: Fallback to original substring match
                for kw in keywords:
                    found = next((c for c in target_df.columns if kw in c.lower()), None)
                    if found: return found
                return None

            brand_col = get_col(['product', 'brand', 'item', 'name'], exclude=['id', 'code', 'number'])
            date_col = get_col(['date', 'time', 'day'], exclude=['month'])
            payment_col = get_col(['payment', 'method', 'type'], exclude=['id'])
            price_col = get_col(['price', 'unit price', 'cost', 'amount'], exclude=['id', 'code'])
            qty_col = get_col(['qty', 'quantity', 'units', 'count'], exclude=['id'])

            # Plotting & Insight functions (Simplified for brevity but maintaining logic)
            def plot_best_selling(parent):
                if not brand_col: raise ValueError("No Product column")
                # target_df is already unencoded
                top_10 = target_df[brand_col].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.barplot(x=top_10.values, y=[str(x)[:15] for x in top_10.index], ax=ax, palette="Blues_r")
                ax.set_title(f"Top 10 {brand_col}"); ax.tick_params(labelsize=8)
                plt.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=parent)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
                return top_10

            def insight_best_selling(parent, data):
                top_name, share = str(data.index[0]), (data.values[0]/len(target_df))*100
                tk.Label(parent, text=top_name[:25], font=("Times New Roman", 14, "bold"), fg="white", bg="#0f0f0f").pack(anchor="w", padx=15)
                tk.Label(parent, text=f"Market Share: {share:.1f}%", font=("Times New Roman", 10), fg=GOLD, bg="#0f0f0f").pack(anchor="w", padx=15, pady=(0,10))
                tk.Message(parent, text=f"• This product is your market leader.\n• It contributes significantly to volume with {data.values[0]} units sold.\n• High turnover indicates strong customer loyalty.", font=("Times New Roman", 9), fg="#ccc", bg="#0f0f0f", width=320).pack(anchor="w", padx=15)
                # REVIEW SECTION
                tk.Label(parent, text="STRATEGIC REVIEW", font=("Times New Roman", 10, "bold", "underline"), fg=GOLD, bg="#0f0f0f").pack(anchor="w", padx=15, pady=(20, 5))
                tk.Message(parent, text="Capitalize on this item's popularity. Ensure zero stock-outs to maintain momentum. Consider creating premium bundles with this item as the anchor to drive sales of slower-moving goods.", font=("Times New Roman", 9, "italic"), fg="#aaa", bg="#0f0f0f", width=320).pack(anchor="w", padx=15)

            def plot_monthly(parent):
                if not date_col: raise ValueError("No Date column")
                temp = target_df.copy()
                temp[date_col] = pd.to_datetime(temp[date_col], errors='coerce')
                monthly = temp[date_col].dt.strftime('%b-%Y').value_counts().sort_index().iloc[-12:]
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(monthly.index, monthly.values, marker='o')
                ax.fill_between(monthly.index, monthly.values, alpha=0.2)
                ax.tick_params(rotation=45, labelsize=8); ax.set_title("Monthly Trend")
                plt.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=parent)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
                return monthly

            def insight_monthly(parent, data):
                peak, low = data.idxmax(), data.idxmin()
                tk.Label(parent, text=f"Peak: {peak}", font=("Times New Roman", 14, "bold"), fg="white", bg="#0f0f0f").pack(anchor="w", padx=15)
                tk.Message(parent, text=f"• Sales volume peaked in {peak}, indicating a strong seasonal preference.\n• Lowest activity was observed in {low}, suggesting a seasonal dip.", font=("Times New Roman", 9), fg="#ccc", bg="#0f0f0f", width=320).pack(anchor="w", padx=15, pady=10)
                # REVIEW SECTION
                tk.Label(parent, text="STRATEGIC REVIEW", font=("Times New Roman", 10, "bold", "underline"), fg=GOLD, bg="#0f0f0f").pack(anchor="w", padx=15, pady=(20, 5))
                tk.Message(parent, text="Plan inventory accumulation 2 months before the peak season. For the low season, consider running clearance sales or special marketing campaigns to boost traffic and cash flow.", font=("Times New Roman", 9, "italic"), fg="#aaa", bg="#0f0f0f", width=320).pack(anchor="w", padx=15)

            def plot_payment(parent):
                if not payment_col: raise ValueError("No Payment column")
                # Filter out nulls/N/A explicitly before counting
                valid_data = target_df[payment_col].dropna()
                # Also filter out string representations of N/A if any remain
                if pd.api.types.is_string_dtype(valid_data):
                    valid_data = valid_data[~valid_data.astype(str).str.lower().isin(['nan', 'n/a', 'none', 'null', 'na'])]
                
                counts = valid_data.value_counts()
                
                # Logic to handle too many categories (Top 5 + Others)
                if len(counts) > 5:
                    top_5 = counts.head(5)
                    others = pd.Series([counts.iloc[5:].sum()], index=["Others"])
                    counts = pd.concat([top_5, others])
                
                # Use plt.subplots for consistency and to avoid GC issues with raw Figure
                fig, ax = plt.subplots(figsize=(6, 3.5))
                
                ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
                
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                canvas = FigureCanvasTkAgg(fig, master=parent)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
                return counts

            def insight_payment(parent, data):
                if data is None or data.empty:
                    tk.Label(parent, text="No Data Available for Insights", font=("Times New Roman", 10), fg="gray", bg="#0f0f0f").pack(anchor="w", padx=15)
                    return

                top = data.index[0]
                tk.Label(parent, text=f"Preferred: {top}", font=("Times New Roman", 14, "bold"), fg="white", bg="#0f0f0f").pack(anchor="w", padx=15)
                tk.Message(parent, text=f"• {top} is the dominant payment method, chosen by the majority.\n• This reflects high trust in this specific channel.", font=("Times New Roman", 9), fg="#ccc", bg="#0f0f0f", width=320).pack(anchor="w", padx=15, pady=10)
                # REVIEW SECTION
                tk.Label(parent, text="STRATEGIC REVIEW", font=("Times New Roman", 10, "bold", "underline"), fg=GOLD, bg="#0f0f0f").pack(anchor="w", padx=15, pady=(20, 5))
                tk.Message(parent, text=f"Ensure your checkout process for {top} is seamless. For minimal share methods, evaluate if the transaction fees justify their maintenance. Consider incentives to shift users to lower-fee channels.", font=("Times New Roman", 9, "italic"), fg="#aaa", bg="#0f0f0f", width=320).pack(anchor="w", padx=15)

                # CHART INSIGHTS REVIEW SECTION
                tk.Label(parent, text="VISUAL ANALYSIS", font=("Times New Roman", 10, "bold", "underline"), fg=GOLD, bg="#0f0f0f").pack(anchor="w", padx=15, pady=(20, 5))
                tk.Message(parent, text="The chart clearly highlights the disparity in payment preferences. A highly fragmented chart suggests a diverse customer base, while a single dominant slice suggests specific demographic habits.", font=("Times New Roman", 9, "italic"), fg="#aaa", bg="#0f0f0f", width=320).pack(anchor="w", padx=15)

            def plot_pq(parent):
                if not (price_col and qty_col): raise ValueError("Missing Price/Qty")
                fig, ax = plt.subplots(figsize=(6, 3.5))
                samp = target_df.sample(min(500, len(target_df)))
                sns.scatterplot(x=samp[price_col], y=samp[qty_col], ax=ax, alpha=0.6)
                ax.set_xlabel("Price"); ax.set_ylabel("Qty")
                canvas = FigureCanvasTkAgg(fig, master=parent)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
                return 0 # Placeholder stat

            def insight_pq(parent, _):
                tk.Label(parent, text="Price Elasticity", font=("Times New Roman", 14, "bold"), fg="white", bg="#0f0f0f").pack(anchor="w", padx=15)
                tk.Message(parent, text="• This scatter plot reveals the relationship between unit price and volume sold.\n• Clusters indicate popular price points where demand is highest.", font=("Times New Roman", 9), fg="#ccc", bg="#0f0f0f", width=320).pack(anchor="w", padx=15, pady=10)
                # REVIEW SECTION
                tk.Label(parent, text="STRATEGIC REVIEW", font=("Times New Roman", 10, "bold", "underline"), fg=GOLD, bg="#0f0f0f").pack(anchor="w", padx=15, pady=(20, 5))
                tk.Message(parent, text="Analyze if higher prices strictly correlate with lower volume. If you see high volume at high prices, you have a premium inelastic product. Use this data to set optimal pricing without sacrificing revenue.", font=("Times New Roman", 9, "italic"), fg="#aaa", bg="#0f0f0f", width=320).pack(anchor="w", padx=15)

            def plot_stock(parent):
                if not brand_col: raise ValueError("No Product column")
                # target_df is already unencoded
                bot_10 = target_df[brand_col].value_counts().tail(10).sort_values()
                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.barplot(x=bot_10.values, y=[str(x)[:15] for x in bot_10.index], ax=ax, palette="Reds")
                ax.set_title("Lowest Selling"); ax.tick_params(labelsize=8)
                plt.tight_layout()
                canvas = FigureCanvasTkAgg(fig, master=parent)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
                return bot_10

            def insight_stock(parent, data):
                worst = str(data.index[0])
                tk.Label(parent, text=f"Lagging: {worst}", font=("Times New Roman", 14, "bold"), fg="red", bg="#0f0f0f").pack(anchor="w", padx=15)
                tk.Message(parent, text=f"• '{worst}' is currently your worst performing item by volume.\n• It ties up working capital with minimal return.", font=("Times New Roman", 9), fg="#ccc", bg="#0f0f0f", width=320).pack(anchor="w", padx=15, pady=10)
                # REVIEW SECTION
                tk.Label(parent, text="STRATEGIC REVIEW", font=("Times New Roman", 10, "bold", "underline"), fg=GOLD, bg="#0f0f0f").pack(anchor="w", padx=15, pady=(20, 5))
                tk.Message(parent, text="Immediate action required: 1. Discount to clear shelf space. 2. Bundle with a best-seller. 3. Investigate if the product is hidden or poorly displayed. Do not reorder until root cause is identified.", font=("Times New Roman", 9, "italic"), fg="#aaa", bg="#0f0f0f", width=320).pack(anchor="w", padx=15)

            create_section("1. Best Selling Products", plot_best_selling, insight_best_selling)
            create_section("2. Best Performing Months", plot_monthly, insight_monthly)
            create_section("3. Customer Payment Preferences", plot_payment, insight_payment)
            create_section("4. Price vs Quantity Analysis", plot_pq, insight_pq)
            create_section("5. Stock Clearance Candidates", plot_stock, insight_stock)
            tk.Label(scrollable_frame, text="-- End of Insights --", fg="gray", bg=BLACK).pack(pady=30)


        def show_statistics(parent):
            # Use wrap for scrollable stats
            scroll_frame = create_scrollable_container(parent)
            
            tk.Label(scroll_frame, text="Statistical Summary (Numeric Columns)", font=("Times New Roman", 16, "bold"), fg=GOLD, bg=BLACK).pack(pady=20, padx=10, anchor="w")
            
            desc = target_df.describe().T.reset_index()
            desc = desc.rename(columns={'index': 'Feature', 'count': 'Cnt', 'mean': 'Mean', 
                                      'std': 'Std', 'min': 'Min', 'max': 'Max', 
                                      '25%': '25%', '50%': 'Med', '75%': '75%'})
            cols = list(desc.columns)

            # Use Treeview but pack it inside the scrollable frame for height control? 
            # Actually standard Treeview has its own scroll, but user asked for consistency and ease.
            # Let's put the treeview in a frame that fills the scrollable area.
            
            tree_frame = tk.Frame(scroll_frame, bg=BLACK)
            tree_frame.pack(fill="both", expand=True, padx=10)
            
            # Since we are already in a scrollable canvas, a treeview inside might glitch (double scroll).
            # Better to just let treeview fill the height if not too huge.
            # But "make charts scrollable" suggests long vertical layouts.
            # For stats table, standard treeview scroll is fine.
            
            style = ttk.Style()
            style.configure("EDA.Treeview", background="#0f0f0f", foreground="white", fieldbackground="#0f0f0f", rowheight=30, borderwidth=0)
            style.configure("EDA.Treeview.Heading", background="#333", foreground="white", font=("Times New Roman", 10, "bold"), relief="flat")
            style.map("EDA.Treeview.Heading", background=[('active', '#444')])
            
            tree = ttk.Treeview(tree_frame, columns=cols, show="headings", style="EDA.Treeview", height=20)
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c, width=80, anchor="center")
            
            for _, row in desc.iterrows():
                vals = [f"{v:.2f}" if isinstance(v, float) else v for v in row]
                tree.insert("", "end", values=vals)
                
            tree.pack(fill="both", expand=True)


        def show_correlation(parent):
            # Wrap in scrollable for consistency even if single chart
            scroll_frame = create_scrollable_container(parent)
            tk.Label(scroll_frame, text="Correlation Heatmap", font=("Times New Roman", 16, "bold"), fg=GOLD, bg=BLACK).pack(pady=20, padx=10, anchor="w")
            
            num = target_df.select_dtypes(include=[np.number])
            if num.empty: return
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap='YlOrBr', ax=ax, cbar_kws={'label': 'Correlation'})
            ax.tick_params(rotation=45)
            canvas = FigureCanvasTkAgg(fig, master=scroll_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)


        def show_distribution(parent):
            scroll_frame = create_scrollable_container(parent)
            tk.Label(scroll_frame, text="Distribution Analysis (KDE Plots)", font=("Times New Roman", 16, "bold"), fg=GOLD, bg=BLACK).pack(pady=20, padx=10, anchor="w")
            
            num_cols = target_df.select_dtypes(include=[np.number]).columns
            
            # Create a BIG vertical grid of plots so it's "scrollable and easily visible"
            # Instead of subplots, let's stack them 1 per row or 2 per row with large height
            
            grid_frame = tk.Frame(scroll_frame, bg=BLACK)
            grid_frame.pack(fill="both", expand=True, padx=10)
            
            plot_types = ['kde', 'hist', 'box']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

            for i, col in enumerate(num_cols):
                # Individual plot for maximum visibility
                plot_frame = tk.Frame(grid_frame, bg="#0f0f0f", bd=1, relief="solid", highlightbackground="#333", pady=10)
                plot_frame.pack(fill="x", pady=10)
                
                tk.Label(plot_frame, text=f"{col} Distribution", font=("Times New Roman", 12, "bold"), fg="white", bg="#0f0f0f").pack()
                
                fig, ax = plt.subplots(figsize=(8, 3))
                
                # Varied plot types
                ptype = plot_types[i % len(plot_types)]
                color = colors[i % len(colors)]
                
                if ptype == 'kde':
                    sns.kdeplot(target_df[col], ax=ax, fill=True, alpha=0.3, color=color)
                    ax.set_title(f"Kernel Density Estimate - {col}", fontsize=10, color='gray')
                elif ptype == 'hist':
                    sns.histplot(target_df[col], ax=ax, kde=False, color=color, alpha=0.6)
                    ax.set_title(f"Histogram - {col}", fontsize=10, color='gray')
                else:
                    sns.boxplot(x=target_df[col], ax=ax, color=color)
                    ax.set_title(f"Box Plot - {col}", fontsize=10, color='gray')

                ax.set_ylabel(""); ax.set_xlabel("")
                
                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
            
            tk.Label(scroll_frame, text="-- End of Distributions --", fg="gray", bg=BLACK).pack(pady=30)

        switch_content("business")
     # ============== MODEL TRAINING MODULE ==============
    def model_training(self):
        if self.current_df is None:
            messagebox.showwarning("Warning", "Please load and preprocess a dataset first!")
            return
        
        # Use encoded_df for model training
        if self.encoded_df is None:
            messagebox.showwarning("Warning", "Please encode categorical variables first in Data Preprocessing!")
            return
        
        self.current_page = "model_training"
        self.clear()
        
        # Main frame
        self.model_frame = tk.Frame(self.root, bg=BLACK)
        self.model_frame.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Title
        tk.Label(self.model_frame, text="ML Model Training & Evaluation", 
                 font=FONT_TITLE, fg=GOLD, bg=BLACK, anchor="w").pack(pady=(20, 20), padx=40)
        
        # Main content container
        content = tk.Frame(self.model_frame, bg=BLACK)
        content.pack(fill='both', expand=True, padx=40, pady=10)
        
        # Left Panel: Configuration (30%)
        left_panel = tk.Frame(content, bg=BLACK)
        left_panel.place(relx=0, rely=0, relwidth=0.30, relheight=1)
        
        # Right Panel: Logs & Results (68%)
        right_panel = tk.Frame(content, bg=BLACK)
        right_panel.place(relx=0.32, rely=0, relwidth=0.68, relheight=1)
        
        # --- LEFT PANEL CONTENT ---
        
        # Header Row: "Configuration" and Back Button
        config_header = tk.Frame(left_panel, bg=BLACK)
        config_header.pack(fill='x', pady=(0, 20))
        
        tk.Label(config_header, text="Configuration", 
                 font=FONT_SUB, fg="white", bg=BLACK).pack(side='left')
        
        # Back Button
        tk.Button(config_header, text="← Back", command=self.feature_page,
                 font=("Segoe UI", 10), bg="#333", fg="white", relief="flat",
                 activebackground="#444", activeforeground="white", padx=10).pack(side='right')
        
        # Target Variable
        tk.Label(left_panel, text="Select Target Variable:", 
                 font=("Segoe UI", 11), fg="#AAAAAA", bg=BLACK, anchor="w").pack(fill='x', pady=(10, 5))
        
        self.target_var = tk.StringVar()
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox", fieldbackground="#333333", background="#333333", 
                        foreground="white", arrowcolor="white", bordercolor="#555555")
        
        tk.Label(left_panel, text="Step 1: Choose Target", 
                 font=("Segoe UI", 14, "bold"), fg=GOLD, bg=BLACK, anchor="w").pack(fill='x', pady=(0, 5))
        tk.Label(left_panel, text="What do you want to predict?", 
                 font=("Segoe UI", 11), fg="#AAAAAA", bg=BLACK, anchor="w").pack(fill='x', pady=(0, 5))

        self.target_combo = ttk.Combobox(left_panel, textvariable=self.target_var, 
                                         state="readonly", font=("Segoe UI", 11))
        self.target_combo.pack(fill='x', pady=(0, 20), ipady=4)
        
        # Populate with numeric columns
        numeric_cols = self.encoded_df.select_dtypes(include=[np.number]).columns.tolist()
        self.target_combo['values'] = numeric_cols
        if numeric_cols:
            if 'TotalSale' in numeric_cols:
                self.target_var.set('TotalSale')
            else:
                self.target_var.set(numeric_cols[-1])
            
        # Algorithms Selection
        tk.Label(left_panel, text="Step 2: Start Analysis", 
                 font=("Segoe UI", 14, "bold"), fg=GOLD, bg=BLACK, anchor="w").pack(fill='x', pady=(20, 5))
        tk.Label(left_panel, text="Select the analysis methods to run:", 
                 font=("Segoe UI", 11), fg="#AAAAAA", bg=BLACK, anchor="w").pack(fill='x', pady=(0, 10))
        
        self.algo_vars = {}
        
        def create_check(parent, text, var_key, default=False):
            self.algo_vars[var_key] = tk.BooleanVar(value=default)
            cb = tk.Checkbutton(parent, text=text, variable=self.algo_vars[var_key],
                               bg=BLACK, fg="#DDDDDD", selectcolor=BLACK, activebackground=BLACK,
                               activeforeground="white", font=("Segoe UI", 11), anchor='w',
                               padx=0, pady=5)
            cb.pack(fill='x')
            return cb

        create_check(left_panel, "Linear Regression", "linear", True)
        create_check(left_panel, "Random Forest", "random_forest", True)

        # --- MOVED: Start Button & Status (Pack at Bottom First) ---
        # Ready Status
        self.status_label = tk.Label(left_panel, text="Ready", 
                                    font=("Segoe UI", 10), fg="#666666", bg=BLACK)
        self.status_label.pack(side='bottom', pady=(0, 20))

        # Start Button
        start_btn = tk.Button(left_panel, text="🚀 Start Training", 
                              font=("Segoe UI", 14, "bold"), bg=GOLD, fg="black",
                              activebackground="#b39226", activeforeground="black",
                              relief="flat", cursor="hand2", command=self.train_models)
        start_btn.pack(side='bottom', fill='x', pady=(10, 10), padx=20, ipady=10)

        # --- REMOVED FEATURE SELECTION UI PER USER REQUEST ---
        # Features will be auto-selected based on numeric type

        self.problem_type = tk.StringVar(value="Auto-Detect")
        
        # --- RIGHT PANEL CONTENT ---
        
        # 1. Logs
        tk.Label(right_panel, text="Training Logs & Console", 
                 font=("Segoe UI", 14), fg="#DDDDDD", bg=BLACK, anchor="w").pack(fill='x', pady=(0, 10))
        
        log_container = tk.Frame(right_panel, bg="black", highlightbackground="#444", highlightthickness=1)
        log_container.place(relx=0, rely=0.08, relwidth=1, relheight=0.40)
        
        log_scrollbar = ttk.Scrollbar(log_container)
        log_scrollbar.pack(side='right', fill='y')
        
        self.log_text = tk.Text(log_container, bg="black", fg="#CCCCCC", font=("Consolas", 10),
                               insertbackground="white", selectbackground="#444",
                               bd=0, highlightthickness=0, yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        log_scrollbar.config(command=self.log_text.yview)
        
        # 2. Results
        results_header_y = 0.52
        tk.Label(right_panel, text="Step 3: View Analysis Results", 
                 font=("Segoe UI", 14, "bold"), fg=GOLD, bg=BLACK, anchor="w").place(relx=0, rely=results_header_y, relwidth=1)
        
        results_container = tk.Frame(right_panel, bg="black", highlightbackground="#444", highlightthickness=1)
        results_container.place(relx=0, rely=results_header_y + 0.08, relwidth=1, relheight=0.40)
        
        style.configure("Treeview", background="black", foreground="white", fieldbackground="black", rowheight=30, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#333", foreground="white", font=("Segoe UI", 10, "bold"), relief="flat")
        
        columns = ("Model", "Algorithm", "R2_Accuracy", "MAE_F1", "Status")
        self.results_table = ttk.Treeview(results_container, columns=columns, show='headings', selectmode='browse')
        
        self.results_table.heading("Model", text="Model Name")
        self.results_table.heading("Algorithm", text="Method Used")
        self.results_table.heading("R2_Accuracy", text="Confidence Score")
        self.results_table.heading("MAE_F1", text="Avg. Error Margin")
        self.results_table.heading("Status", text="Status")
        
        self.results_table.column("Model", width=150)
        self.results_table.column("Algorithm", width=200)
        self.results_table.column("R2_Accuracy", width=120, anchor='center')
        self.results_table.column("MAE_F1", width=120, anchor='center')
        self.results_table.column("Status", width=100, anchor='center')
        
        table_scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=self.results_table.yview)
        self.results_table.configure(yscrollcommand=table_scrollbar.set)
        
        self.results_table.pack(side='left', fill='both', expand=True)
        table_scrollbar.pack(side='right', fill='y')

    def train_models(self):
        """Train selected machine learning models based on user selection"""
        if self.encoded_df is None:
            messagebox.showwarning("Warning", "Please encode categorical variables first!")
            return
        
        target = self.target_var.get()
        if not target:
            messagebox.showwarning("Warning", "Please select a target variable!")
            return
        
        # Enable log text
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.insert('end', "="*60 + "\n")
        self.log_text.insert('end', f" STARTED TRAINING JOB: {pd.Timestamp.now()}\n")
        self.log_text.insert('end', "="*60 + "\n\n")
        self.log_text.insert('end', f"Target Variable: {target}\n")
        
        # Prepare data
        try:
            # AUTO-SELECT features (All numeric columns except target)
            X = self.encoded_df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
            selected_features = X.columns.tolist()
            
            if not selected_features:
                 messagebox.showwarning("Warning", "No numeric features found from the dataset (after exclusion).")
                 return

            self.log_text.insert('end', f"Selected Features (Auto): {selected_features}\n")
            
            # --- DEBUG INFO ---
            self.log_text.insert('end', f"Data Shape: {X.shape}\n")
            self.log_text.insert('end', "Data Types:\n" + str(X.dtypes) + "\n")
            
            # Sanitize X: Replace Infinite with NaN
            # Sklearn's SimpleImputer handles NaN but not Infinity by default in some versions or configurations
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Check for All-NaN columns
            nan_cols = X.columns[X.isna().all()].tolist()
            if nan_cols:
                 self.log_text.insert('end', f"Warning: Dropping ALL-NaN columns: {nan_cols}\n")
                 X = X.drop(columns=nan_cols)
                 
            if X.empty:
                 messagebox.showerror("Error", "All feature columns were dropped (empty or all-NaN).")
                 self.log_text.config(state='disabled')
                 return

            y = self.encoded_df[target]
            
            # Sanitize y
            if pd.api.types.is_numeric_dtype(y):
                y = y.replace([np.inf, -np.inf], np.nan)
                # Drop rows where target is NaN (we can't train on missing target)
                mask = y.notna()
                X = X[mask]
                y = y[mask]
                self.log_text.insert('end', f"Rows remaining after cleaning Target NaNs: {len(y)}\n")
            
            if len(y) == 0:
                 messagebox.showerror("Error", "No valid data rows left after cleaning.")
                 return

            
            # Use X as is (already numeric checked in listbox population)
            if X.empty:
                self.log_text.insert('end', "Error: No data found for selected features!\n")
                self.log_text.config(state='disabled')
                return
                
            y = self.encoded_df[target]
            
            # (No explicit dropping logic here as we use user selection directly)
                
        except Exception as e:
            self.log_text.insert('end', f"Error preparing data: {str(e)}\n")
            self.log_text.config(state='disabled')
            return
        
        # Determine problem type
        problem = self.problem_type.get()
        if problem == "Auto-Detect":
            if y.dtype in ['int64', 'float64', 'int32', 'float32']:
                problem = "Regression" # Always default to Regression for numeric targets to allow Forecasting
            else:
                problem = "Classification"
            self.log_text.insert('end', f"Problem Type Detected: {problem}\n")
        else:
            self.log_text.insert('end', f"Problem Type Selected: {problem}\n")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Handle missing values (NaN) with mean imputation
        n_nans_before = X_train.isna().sum().sum()
        self.log_text.insert('end', f"DEBUG: NaNs before imputation: {n_nans_before}\n")
        
        try:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            # Fit on training data, transform both
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
            
            # Reconstruct DataFrames
            X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
            X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)
            
            n_nans_after = X_train.isna().sum().sum()
            self.log_text.insert('end', f"DEBUG: NaNs after imputation: {n_nans_after}\n")
            
            self.log_text.insert('end', "Note: Missing values imputed with mean strategy.\n")
            
        except Exception as e:
            self.log_text.insert('end', f"Warning: Imputation failed: {e}\n")
            messagebox.showerror("Imputation Error", f"Imputation failed: {e}")

        self.log_text.insert('end', f"Train Data: {X_train.shape}\n")
        self.log_text.insert('end', f"Test Data: {X_test.shape}\n\n")
        
        # Determine which algorithms to run
        run_linear = self.algo_vars["linear"].get()
        run_rf = self.algo_vars["random_forest"].get()
        
        try:
            self.log_text.insert('end', f"DEBUG: run_linear={run_linear}, run_rf={run_rf}\n")
        except:
            print(f"DEBUG: run_linear={run_linear}, run_rf={run_rf}")

        if not (run_linear or run_rf):
            messagebox.showwarning("Warning", "Please select at least one algorithm!")
            self.log_text.config(state='disabled')
            return

        # Target Scaling for Regression - DISABLED per user request (Do not standardize target)
        scaler_y = None
        # if problem == "Regression":
        #    ... (removed target scaling)

        models_trained = 0
        self.model_results = {}
        
        # Clear existing table data if any
        for item in self.results_table.get_children():
            self.results_table.delete(item)
        
        # Helper to log and train
        def train_single_model(name, algo_obj, is_baseline=False):
            nonlocal models_trained
            try:
                self.log_text.insert('end', "-"*40 + "\n")
                role = "BASELINE" if is_baseline else "MODEL"
                self.log_text.insert('end', f"Running {name} ({role})...\n")
                self.root.update()
                
                algo_obj.fit(X_train, y_train)
                y_pred = algo_obj.predict(X_test)
                
                # Inverse Transform & FIX: Ensure no negative predictions for Regression
                if problem == "Regression":
                    if scaler_y:
                        # Inverse transform predictions
                        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                        # Use y_test_orig for metrics logic (although y_test is untouched here, let's correspond)
                        # Actually y_test was NOT scaled (only y_train was scaled above?), no wait.
                        # I only scaled y_train? y_test needs to be untouched for metrics?
                        # Ah, y_train was assigned scaled values. y_test is still original.
                        # So y_pred (scaled output) needs inverse transform to match y_test (original). Correct.
                    
                    # STEP 2: Clipping
                    y_pred = np.maximum(y_pred, 0)
                
                # Metrics
                if problem == "Regression":
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    self.log_text.insert('end', f" > R2 Score: {r2:.4f}\n")
                    self.log_text.insert('end', f" > MAE: {mae:.4f}\n")
                    score_display = f"{r2:.4f}"
                    metric_display = f"{mae:.4f}"
                    type_str = "Regression"
                else:
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    self.log_text.insert('end', f" > Accuracy: {acc:.4f}\n")
                    self.log_text.insert('end', f" > F1 Score: {f1:.4f}\n")
                    score_display = f"{acc:.4f}"
                    metric_display = f"{f1:.4f}"
                    type_str = "Classification"
                    
                # Store
                key = f"{target} - {name}"
                self.model_results[key] = {
                    'Target': target,
                    'Type': type_str,
                    'Algo': name,
                    'R2_Acc': score_display,
                    'MAE_F1': metric_display,
                    'model': algo_obj,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'X_test': X_test,
                    'scaler_y': scaler_y
                }
                
                if hasattr(algo_obj, 'feature_importances_'):
                    self.model_results[key]['feature_importance'] = algo_obj.feature_importances_
                
                models_trained += 1
                self.log_text.insert('end', " > Status: COMPLETED\n")
                
                # Update Table
                self.results_table.insert("", "end", values=(key, name, score_display, metric_display, "Completed"))
                
            except Exception as e:
                self.log_text.insert('end', f" > Error: {str(e)}\n")
                self.results_table.insert("", "end", values=(name, name, "Error", "Error", "Failed"))

        # 1. Linear/Logistic (Baseline)
        if run_linear:
            if problem == "Regression":
                train_single_model("Linear Regression", LinearRegression(), is_baseline=True)
            else:
                train_single_model("Logistic Regression", LogisticRegression(max_iter=1000), is_baseline=True)
                
        # 2. Random Forest (Main)
        if run_rf:
            if problem == "Regression":
                train_single_model("Random Forest Regressor", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
            else:
                train_single_model("Random Forest Classifier", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
        
        
        self.log_text.insert('end', "\n" + "="*60 + "\n")
        self.log_text.insert('end', " ALL JOBS FINISHED.\n")
        self.log_text.insert('end', f" Total Models Trained: {models_trained}\n")
        self.log_text.insert('end', "="*60 + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        
        if models_trained > 0:
            messagebox.showinfo("Success", "Training Complete!")
            # Update status label
            self.status_label.config(text=f"Training Completed. {models_trained} models ready.", fg=GREEN)
        

    # ============== ML MODEL RESULTS & ANALYSIS (Redesigned) ==============
    def results_analysis(self):
        # Check if models have been trained
        if not hasattr(self, 'model_results') or not self.model_results:
            messagebox.showinfo("Info", "No model results available. Please train models first.")
            self.model_training()
            return

        self.current_page = "results"
        self.clear()

        # Main Container
        self.results_frame = tk.Frame(self.root, bg=BLACK)
        self.results_frame.place(x=0, y=0, relwidth=1, relheight=1)

        # Title
        tk.Label(self.results_frame, text="ML Model Results & Analysis",
                 font=FONT_TITLE, fg=GOLD, bg=BLACK, anchor="w").pack(pady=(20, 10), padx=40, fill='x')

        # Split Content
        content = tk.Frame(self.results_frame, bg=BLACK)
        content.pack(fill='both', expand=True, padx=40, pady=(0, 20))

        # --- LEFT PANEL: KPI & METRICS (35%) ---
        left_panel = tk.Frame(content, bg=CARD, bd=1, relief="flat")
        left_panel.place(relx=0, rely=0, relwidth=0.35, relheight=1)

        tk.Label(left_panel, text="Key Performance Indicators", 
                font=FONT_SUB, fg="white", bg=CARD, anchor="w").pack(pady=20, padx=20, fill='x')

        # Target Variable / Model Selection
        tk.Label(left_panel, text="Select Model Result:", 
                font=("Times New Roman", 10), fg="gray", bg=CARD, anchor="w").pack(padx=20, pady=(10, 2), fill='x')
        
        target_frame = tk.Frame(left_panel, bg="#2C2C2C", bd=1, relief="flat")
        target_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        # Configure style for dark combobox
        style = ttk.Style()
        style.configure("Dark.TCombobox", fieldbackground="#2C2C2C", background="#333", foreground="white", arrowcolor="gold")
        style.map("Dark.TCombobox", fieldbackground=[("readonly", "#2C2C2C")], foreground=[("readonly", "white")])

        self.res_combo_var = tk.StringVar()
        self.res_combo = ttk.Combobox(target_frame, textvariable=self.res_combo_var, 
                                     state="readonly", font=("Times New Roman", 10), style="Dark.TCombobox")
        self.res_combo.pack(fill='x', padx=5, pady=5)
        
        # Populate Combobox with trained models
        if self.model_results:
            self.res_combo['values'] = list(self.model_results.keys())
            self.res_combo.current(0)  # Select first model
        

        
        # ================== UPDATED SECTION: KPI METRICS WITH SCROLLBARS ==================
        # ================== UPDATED SECTION: KPI METRICS WITH SCROLLBARS ==================
        # Add future prediction controls
        tk.Label(left_panel, text="Future Sales Prediction:", 
                font=("Times New Roman", 11, "bold"), fg=GOLD, bg=CARD, anchor="w").pack(padx=20, pady=(20, 5), fill='x')
        
        # Prediction days input
        pred_frame = tk.Frame(left_panel, bg=CARD)
        pred_frame.pack(padx=20, pady=(5, 10), fill='x')
        
        tk.Label(pred_frame, text="Forecast Days:", 
                font=("Times New Roman", 10), fg="white", bg=CARD, anchor="w").pack(side='left')
        
        self.forecast_days = tk.StringVar(value="30")
        days_entry = tk.Entry(pred_frame, textvariable=self.forecast_days, 
                             font=("Times New Roman", 10), width=8, bg="#1A1A1A", fg="white")
        days_entry.pack(side='left', padx=(10, 0))
        
        # Confidence interval
        conf_frame = tk.Frame(left_panel, bg=CARD)
        conf_frame.pack(padx=20, pady=(5, 10), fill='x')
        
        tk.Label(conf_frame, text="Confidence Level:", 
                font=("Times New Roman", 10), fg="white", bg=CARD, anchor="w").pack(side='left')
        
        self.confidence_level = tk.StringVar(value="95")
        conf_entry = tk.Entry(conf_frame, textvariable=self.confidence_level, 
                             font=("Times New Roman", 10), width=8, bg="#1A1A1A", fg="white")
        conf_entry.pack(side='left', padx=(10, 0))
        tk.Label(conf_frame, text="%", 
                font=("Times New Roman", 10), fg="white", bg=CARD, anchor="w").pack(side='left', padx=(5, 0))
        
        # Prediction button
        self.predict_btn = tk.Button(left_panel, text="📊 Generate Forecast", 
                                    font=("Times New Roman", 11, "bold"),
                                    bg=GREEN, fg="white", 
                                    activebackground="#218838", activeforeground="white",
                                    relief="flat", cursor="hand2", 
                                    command=self.generate_future_predictions)
        self.predict_btn.pack(fill='x', padx=30, pady=15, ipady=8)


        def update_metrics(event=None):
            selection = self.res_combo_var.get()
            if selection and selection in self.model_results:
                m = self.model_results[selection]
                # Update forecast inputs state based on model type
                is_regression = m.get("Type", "") == "Regression"
                if is_regression:
                    self.predict_btn.config(state="normal", bg=GREEN, text="📊 Generate Forecast")
                    # Also enable inputs
                    days_entry.config(state='normal')
                    conf_entry.config(state='normal')
                else:
                    self.predict_btn.config(state="normal", bg=GRAY, text="Forecast (N/A for Classif.)")
                    pass
                


                
                # Also set this as 'latest' for any other logic relying on it
                self.latest_model_metrics = m
        
        self.res_combo.bind("<<ComboboxSelected>>", update_metrics)
        # Initial call
        update_metrics()

        # Generate Plot Button
        def generate_plot():
            selection = self.res_combo_var.get()
            if not selection or selection not in self.model_results:
                messagebox.showinfo("Info", "No model result selected.")
                return
            
            # Clear previous
            for w in self.chart_area.winfo_children(): 
                w.destroy()
            
            m = self.model_results[selection]
            y_test = m['y_test']
            y_pred = m['y_pred']
            task = m['Type']
            algo = m['Algo']
            
            # Create a notebook for multiple plot types
            plot_notebook = ttk.Notebook(self.chart_area)
            plot_notebook.pack(fill='both', expand=True)
            
            # Tab 1: Actual vs Predicted / Confusion Matrix
            tab1 = tk.Frame(plot_notebook, bg="#151515")
            plot_notebook.add(tab1, text="Predictions")
            
            # Tab 2: Feature Importance (if available)
            tab2 = tk.Frame(plot_notebook, bg="#151515")
            plot_notebook.add(tab2, text="Feature Importance")
            
            # Tab 3: Residuals/Error Analysis
            tab3 = tk.Frame(plot_notebook, bg="#151515")
            plot_notebook.add(tab3, text="Error Analysis")
            
            # Tab 4: Future Sales Forecast (only for regression)
            if task == "Regression":
                tab4 = tk.Frame(plot_notebook, bg="#151515")
                plot_notebook.add(tab4, text="Future Forecast")
            
            # Tab 1: Prediction Visualization
            if task == "Regression":
                fig1, ax1 = plt.subplots(figsize=(5, 4))
                fig1.patch.set_facecolor('#1A1A1A')
                ax1.set_facecolor('#1A1A1A')
                
                # Actual vs Predicted scatter plot
                ax1.scatter(y_test, y_pred, color=GOLD, alpha=0.6, edgecolors='w', s=50)
                
                # Line of perfect fit
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
                
                # Add regression line
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
                line_x = np.array([min_val, max_val])
                line_y = slope * line_x + intercept
                ax1.plot(line_x, line_y, 'b-', lw=2, alpha=0.7, label=f'Fit (R²={r_value**2:.3f})')
                
                ax1.set_xlabel("Actual Values", color="white", fontsize=10)
                ax1.set_ylabel("Predicted Values", color="white", fontsize=10)
                ax1.set_title(f"Actual vs Predicted ({algo})", color=GOLD, fontsize=12, pad=15)
                ax1.tick_params(colors="white")
                ax1.legend(facecolor='#1A1A1A', edgecolor='white', labelcolor='white', fontsize=9)
                ax1.grid(True, alpha=0.2, color='gray')
                
                for s in ax1.spines.values(): 
                    s.set_edgecolor('#333')
                
                canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
                canvas1.draw()
                canvas1.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
                
            else:  # Classification
                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                fig1.patch.set_facecolor('#1A1A1A')
                ax1.set_facecolor('#1A1A1A')
                ax2.set_facecolor('#1A1A1A')
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax1, cbar=True, 
                           cbar_kws={'label': 'Count'})
                ax1.set_title(f"Confusion Matrix ({algo})", color='white', pad=15)
                ax1.set_xlabel("Predicted", color='white')
                ax1.set_ylabel("Actual", color='white')
                ax1.tick_params(colors='white')
                
                # Classification Report (simplified as bar chart)
                class_report = classification_report(y_test, y_pred, output_dict=True)
                metrics_names = ['precision', 'recall', 'f1-score']
                metrics_values = [class_report['weighted avg'][m] for m in metrics_names]
                
                bars = ax2.bar(metrics_names, metrics_values, color=[GOLD, BLUE, GREEN])
                ax2.set_ylim([0, 1])
                ax2.set_title("Weighted Average Metrics", color='white', pad=15)
                ax2.set_ylabel("Score", color='white')
                ax2.tick_params(colors='white')
                
                # Add value labels on bars
                for bar, val in zip(bars, metrics_values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)
                
                for ax in [ax1, ax2]:
                    for s in ax.spines.values(): 
                        s.set_edgecolor('#333')
                
                plt.tight_layout()
                canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
                canvas1.draw()
                canvas1.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
            
            # Tab 2: Feature Importance (if available)
            if "feature_importance" in m or "top_features" in m:
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                fig2.patch.set_facecolor('#1A1A1A')
                ax2.set_facecolor('#1A1A1A')
                
                if "top_features" in m:
                    # Use top features from Gradient Boosting
                    top_features = m["top_features"]
                    feature_names = [feat[0] for feat in top_features]
                    importances = [feat[1] for feat in top_features]
                    
                    bars = ax2.barh(range(len(feature_names)), importances, color=GOLD)
                    ax2.set_yticks(range(len(feature_names)))
                    ax2.set_yticklabels(feature_names)
                    ax2.set_xlabel('Importance', color='white')
                    ax2.set_title(f'Top Feature Importances ({algo})', color='white', pad=15)
                    ax2.tick_params(colors='white')
                    
                    # Add value labels
                    for i, (bar, imp) in enumerate(zip(bars, importances)):
                        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                                f'{imp:.4f}', ha='left', va='center', color='white', fontsize=9)
                
                elif "feature_importance" in m:
                    # Generic feature importance
                    importances = m["feature_importance"]
                    if len(importances) > 10:  # Show top 10 if too many
                        top_indices = np.argsort(importances)[-10:][::-1]
                        importances = importances[top_indices]
                        if "X_test" in m:
                            feature_names = m["X_test"].columns[top_indices]
                        else:
                            feature_names = [f"Feature {i}" for i in top_indices]
                    else:
                        feature_names = [f"Feature {i}" for i in range(len(importances))]
                    
                    bars = ax2.barh(range(len(feature_names)), importances, color=GOLD)
                    ax2.set_yticks(range(len(feature_names)))
                    ax2.set_yticklabels(feature_names)
                    ax2.set_xlabel('Importance', color='white')
                    ax2.set_title(f'Feature Importances ({algo})', color='white', pad=15)
                    ax2.tick_params(colors='white')
                
                for s in ax2.spines.values(): 
                    s.set_edgecolor('#333')
                
                canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
                canvas2.draw()
                canvas2.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
            else:
                tk.Label(tab2, text="Feature importance not available for this model.", 
                        font=("Times New Roman", 11), fg="gray", bg="#151515").pack(pady=50)
            
            # Tab 3: Error Analysis
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            fig3.patch.set_facecolor('#1A1A1A')
            ax3.set_facecolor('#1A1A1A')
            
            if task == "Regression":
                # Residual plot (Absolute values as requested)
                residuals = np.abs(y_test - y_pred)
                ax3.scatter(y_pred, residuals, color=GOLD, alpha=0.6, edgecolors='w', s=50)
                ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
                ax3.set_xlabel("Predicted Values", color="white", fontsize=10)
                ax3.set_ylabel("Absolute Error", color="white", fontsize=10)
                ax3.set_title(f"Absolute Error Plot ({algo})", color=GOLD, fontsize=12, pad=15)
                
                # Add histogram of residuals
                ax3_hist = ax3.inset_axes([0.02, 0.02, 0.3, 0.3])
                ax3_hist.hist(residuals, bins=20, color=GOLD, alpha=0.7, edgecolor='white')
                ax3_hist.set_facecolor('#1A1A1A')
                ax3_hist.tick_params(colors='white', labelsize=8)
                for s in ax3_hist.spines.values(): 
                    s.set_edgecolor('#333')
                
            else:
                # Error rate by class
                unique_classes = np.unique(y_test)
                error_rates = []
                for cls in unique_classes:
                    mask = (y_test == cls)
                    if mask.any():
                        class_errors = (y_pred[mask] != y_test[mask]).sum()
                        error_rate = class_errors / mask.sum()
                        error_rates.append(error_rate)
                    else:
                        error_rates.append(0)
                
                bars = ax3.bar([f"Class {cls}" for cls in unique_classes], error_rates, color=GOLD)
                ax3.set_xlabel("Class", color="white", fontsize=10)
                ax3.set_ylabel("Error Rate", color="white", fontsize=10)
                ax3.set_title(f"Error Rate by Class ({algo})", color=GOLD, fontsize=12, pad=15)
                
                # Add value labels
                for bar, err in zip(bars, error_rates):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{err:.3f}', ha='center', va='bottom', color='white', fontsize=9)
            
            ax3.tick_params(colors="white")
            ax3.grid(True, alpha=0.2, color='gray')
            for s in ax3.spines.values(): 
                s.set_edgecolor('#333')
            
            canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
            
            # Tab 4: Future Forecast Placeholder
            if task == "Regression":
                placeholder_frame = tk.Frame(tab4, bg="#151515")
                placeholder_frame.pack(fill='both', expand=True)
                
                tk.Label(placeholder_frame, text="📈", font=("Times New Roman", 40), fg="#333", bg="#151515").pack(pady=(60, 10))
                tk.Label(placeholder_frame, text="Click 'Generate Forecast' button\n to predict future sales", 
                        font=("Times New Roman", 11), fg="gray", bg="#151515").pack()

        # Button removed as per new UI design

        # --- RIGHT PANEL: CHARTS (63%) ---
        right_panel = tk.Frame(content, bg=CARD, bd=1, relief="flat")
        right_panel.place(relx=0.37, rely=0, relwidth=0.63, relheight=1)

        tk.Label(right_panel, text="Interactive Analysis Dashboard", 
                font=FONT_SUB, fg="white", bg=CARD, anchor="w").pack(pady=20, padx=20, fill='x')

        # Chart Container
        chart_container = tk.Frame(right_panel, bg="#151515", bd=1, relief="solid", 
                                 highlightbackground="#333")
        chart_container.pack(fill='both', expand=True, padx=30, pady=(0, 30))
        
        self.chart_area = tk.Frame(chart_container, bg="#151515")
        self.chart_area.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Placeholder
        tk.Label(self.chart_area, text="📈", font=("Times New Roman", 40), fg="#333", bg="#151515").pack(pady=(60, 10))
        tk.Label(self.chart_area, text="Plots will appear here based\non selected analysis", 
                font=FONT_SMALL, fg="gray", bg="#151515").pack()

        # --- BOTTOM NAV (Footer) ---
        footer = tk.Frame(self.results_frame, bg=BLACK)
        footer.pack(side="bottom", fill="x", padx=40, pady=20)
        
        # Back Button -> Feature Page
        self.premium_button(footer, "Back to Training", 
                           self.model_training, width=20, bg=GOLD, fg=BLACK).pack(side="left", padx=10)
        
        # Save Best Model Button
        self.premium_button(footer, "💾 Save Best Model", 
                           self.save_best_model, width=20, bg=BLUE, fg="white").pack(side="left", padx=10)
        
        # Go to Final Outcome
        self.premium_button(footer, "📋 Outcome", 
                           self.outcome_analysis, width=20, bg=GREEN, fg="white").pack(side="right", padx=10)

    def generate_future_predictions(self):
        """Generate future sales predictions using the selected model"""
        selection = self.res_combo_var.get()
        if not selection or selection not in self.model_results:
            messagebox.showinfo("Info", "No model result selected.")
            return
        
        m = self.model_results[selection]
        
        # Check if it's a regression model
        if m.get("Type") != "Regression":
            messagebox.showwarning("Warning", "Future forecasting is only available for regression models!")
            return
        
        try:
            forecast_days = int(self.forecast_days.get())
            if forecast_days <= 0 or forecast_days > 365:
                raise ValueError("Forecast days should be between 1 and 365")
            
            confidence = float(self.confidence_level.get())
            if confidence < 0 or confidence > 100:
                raise ValueError("Confidence level should be between 0 and 100")
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
            return
        
        # Get the trained model
        model = m['model']
        X_test = m['X_test']

        # --- FIX FOR OUTCOME PAGE FORECASTING ---
        # If we are in the Outcome Analysis page, self.chart_area (from Analysis page) doesn't exist.
        # We must create a temporary Modal Window to display the results.
        if hasattr(self, 'current_page') and self.current_page == "outcome_analysis":
            modal = tk.Toplevel(self.root)
            modal.title(f"Future Sales Forecast - {forecast_days} Days")
            modal.geometry("1100x800")
            modal.configure(bg="#050505")
            
            # Add a close button logic if needed, but Toplevel default is fine.
            # Create a container frame that mimics 'self.chart_area'
            chart_container = tk.Frame(modal, bg="#050505")
            chart_container.pack(fill="both", expand=True)
            
            # Temporarily redirect self.chart_area to this new modal frame
            self.chart_area = chart_container
        # ----------------------------------------
        
        # For time series forecasting, we need historical data
        # Try to find date column in original data
        date_col = None
        for col in self.current_df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col:
            # Time series forecasting
            try:
                # Create time series data
                temp_df = self.current_df.copy()
                temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
                
                # Find the target column
                target = m.get("Target", None)
                if not target:
                    messagebox.showerror("Error", "Target column not found!")
                    return
                
                # Create time series - handle duplicate dates by aggregating
                temp_df = temp_df.sort_values(date_col)
                
                # Check for duplicate dates
                duplicate_dates = temp_df[date_col].duplicated().sum()
                if duplicate_dates > 0:
                    # Aggregate by date (sum or mean depending on context)
                    time_series = temp_df.groupby(date_col)[target].sum()  # or .mean()
                    messagebox.showinfo("Info", 
                                      f"Aggregated {duplicate_dates} duplicate dates by summing target values.")
                else:
                    time_series = temp_df.set_index(date_col)[target]
                
                # Fill missing dates if needed
                # First ensure we have a proper date range
                min_date = time_series.index.min()
                max_date = time_series.index.max()
                
                # Create complete date range
                full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
                
                # Reindex with complete date range
                time_series = time_series.reindex(full_date_range)
                
                # Fill missing values
                time_series = time_series.fillna(time_series.mean())
                
                # For demonstration, using simple moving average forecast
                # In production, you would use ARIMA, Prophet, or LSTM
                window_size = max(1, min(30, len(time_series) // 10))
                
                # Calculate moving average
                moving_avg = time_series.rolling(window=window_size, min_periods=1).mean()
                
                # Simple forecast: extrapolate trend
                last_value = moving_avg.iloc[-1]
                trend = moving_avg.diff().mean()
                
                # Handle NaN trend (e.g. not enough data points)
                if pd.isna(trend):
                    trend = 0
                if pd.isna(last_value):
                    last_value = time_series.mean() if not pd.isna(time_series.mean()) else 0

                # Generate future dates
                last_date = time_series.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                            periods=forecast_days, freq='D')
                
                # Generate predictions
                future_predictions = []
                for i in range(forecast_days):
                    future_predictions.append(last_value + trend * (i + 1))
                
                # FIX: Ensure no negative predictions and handle NaNs
                predictions = np.maximum(future_predictions, 0)
                predictions = np.nan_to_num(predictions, nan=0.0) # Safety net
                
                # Calculate confidence intervals
                std_dev = time_series.std()
                if pd.isna(std_dev):
                    std_dev = 0.0

                # Adjust z-score based on confidence level
                if confidence == 90:
                    z_score = 1.645
                elif confidence == 95:
                    z_score = 1.96
                elif confidence == 99:
                    z_score = 2.576
                else:
                    z_score = 1.96  # default to 95%
                
                # Update bounds to be non-negative
                lower_bound = [max(0, p - z_score * std_dev) for p in predictions]
                upper_bound = [p + z_score * std_dev for p in predictions]
                
                # Update the forecast label
                avg_prediction = np.mean(predictions)

                
                # Plot the forecast
                self.plot_forecast_chart(time_series, moving_avg, future_dates, 
                                       predictions, lower_bound, upper_bound, 
                                       target, forecast_days, confidence)
                
            except Exception as e:
                messagebox.showerror("Forecasting Error", 
                                   f"Time series forecasting failed:\n{str(e)}")
                import traceback
                print(traceback.format_exc())
        else:
            # Non-time series forecasting - generate synthetic future data
            messagebox.showinfo("Info", "No date column found. Using synthetic future data.")
            
            # Generate synthetic future features based on historical patterns
            last_features = X_test.iloc[-1:].copy()
            
            # Create future predictions by slightly modifying last features
            future_predictions = []
            for i in range(forecast_days):
                # Add some randomness to simulate future variation
                synthetic_features = last_features.copy()
                for col in synthetic_features.columns:
                    if pd.api.types.is_numeric_dtype(synthetic_features[col]):
                        noise = np.random.normal(0, 0.05)  # 5% variation
                        synthetic_features[col] = synthetic_features[col] * (1 + noise)
                
                # Predict
                pred = model.predict(synthetic_features)[0]
                
                # Inverse Transform if available
                scaler_y = m.get('scaler_y')
                if scaler_y:
                    pred = scaler_y.inverse_transform([[pred]])[0][0]

                # FIX: Ensure no negative predictions
                if pred < 0:
                    pred = 0
                future_predictions.append(pred)
            
            # Calculate statistics
            avg_prediction = np.mean(future_predictions)

            
            # Create simple line plot for future predictions
            self.plot_simple_forecast(future_predictions, forecast_days, 
                                    m.get("Target", "Sales"))

    def plot_forecast_chart(self, historical, moving_avg, future_dates, 
                           predictions, lower_bound, upper_bound, 
                           target, forecast_days, confidence):
        """Plot the time series forecast with confidence intervals"""
        
        # Clear previous
        for w in self.chart_area.winfo_children(): 
            w.destroy()
        
        # Create notebook if not exists
        plot_notebook = ttk.Notebook(self.chart_area)
        plot_notebook.pack(fill='both', expand=True)
        
        # Tab for forecast
        forecast_tab = tk.Frame(plot_notebook, bg="#151515")
        plot_notebook.add(forecast_tab, text="Future Sales Forecast")
        
        # Forecast Statistics tab removed as per request
        
        # Display forecast statistics in the scrollable frame
        # self.display_forecast_statistics_scrollable call removed
        
        # --- CHART GENERATION ---
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1A1A1A')
        ax.set_facecolor('#1A1A1A')
        
        # Plot historical data - show last 90 days for clarity
        if len(historical) > 90:
            historical_to_plot = historical.iloc[-90:]
            moving_avg_to_plot = moving_avg.iloc[-90:]
        else:
            historical_to_plot = historical
            moving_avg_to_plot = moving_avg
        
        # 1. Historical Data (Blue)
        ax.plot(historical_to_plot.index, historical_to_plot.values, 
                color='blue', linewidth=1.5, alpha=0.7, label='Historical Data')
        
        # 2. Moving Average (Green)
        ax.plot(moving_avg_to_plot.index, moving_avg_to_plot.values, 
                color='lime', linewidth=2, label='Moving Average (Trend)')
        
        # 3. Forecast Start Line (Yellow Dotted)
        forecast_start = future_dates[0]
        ax.axvline(x=forecast_start, color='yellow', linestyle=':', linewidth=2, label='Forecast Start')
        
        # 4. Forecast (Red Dashed)
        ax.plot(future_dates, predictions, color='red', linestyle='--', linewidth=2, label='Forecast')
        
        # 5. Confidence Interval (Red Shaded)
        ax.fill_between(future_dates, lower_bound, upper_bound, color='red', alpha=0.2, label=f'{confidence}% Confidence Interval')
        
        # Annotations
        # Today Marker (Anchored to Forecast Start)
        today_str = forecast_start.strftime('%Y-%m-%d')
        y_limits = ax.get_ylim()
        text_y = y_limits[1] * 0.9  # 90% height
        
        ax.annotate(f'Today: {today_str}', 
                   xy=(forecast_start, text_y), 
                   xytext=(-60, 0), textcoords='offset points',
                   fontsize=10, color='white', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#333", edgecolor="none"),
                   arrowprops=dict(arrowstyle="->", color='white', connectionstyle="arc3,rad=0.2"))
        
        # Summary Box (Bottom Left)
        total_forecast = np.sum(predictions)
        avg_daily = np.mean(predictions)
        peak_idx = np.argmax(predictions)
        peak_value = predictions[peak_idx]
        peak_date = future_dates[peak_idx].strftime('%Y-%m-%d')
        
        summary_text = (
            f"📋 Forecast Summary ({forecast_days} days):\n"
            f"• Total Sales: ₹{total_forecast:,.2f}\n"
            f"• Avg Daily: ₹{avg_daily:,.2f}\n"
            f"• Peak Day: {peak_date} (₹{peak_value:,.2f})"
        )
        
        # Place text box in bottom left
        props = dict(boxstyle='round,pad=0.5', facecolor='#1A1A1A', alpha=0.9, edgecolor='gold')
        ax.text(0.02, 0.03, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', color='white', bbox=props)
        
        # Formatting
        ax.set_title(f"Future Sales Forecast ({forecast_days} Days Ahead)", color=GOLD, fontsize=14, pad=15)
        ax.set_xlabel("Date", color="white", fontsize=10)
        ax.set_ylabel(target, color="white", fontsize=10)
        ax.tick_params(colors="white", labelrotation=45)
        ax.grid(True, alpha=0.1, color='gray')
        
        for s in ax.spines.values(): 
            s.set_edgecolor('#333')
            
        # Legend (Top Left)
        ax.legend(loc='upper left', facecolor='#1A1A1A', edgecolor='gray', labelcolor='white')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=forecast_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def display_forecast_statistics_scrollable(self, parent, predictions, future_dates, 
                                             lower_bound, upper_bound, target, forecast_days, confidence):
        """Display detailed forecast statistics in a scrollable frame"""
        
        # Calculate statistics
        total_forecast = np.sum(predictions)
        avg_daily = np.mean(predictions)
        std_daily = np.std(predictions)
        max_value = np.max(predictions)
        min_value = np.min(predictions)
        max_day = future_dates[np.argmax(predictions)].strftime('%Y-%m-%d')
        min_day = future_dates[np.argmin(predictions)].strftime('%Y-%m-%d')
        
        # Display statistics
        stats_frame = tk.Frame(parent, bg="#1A1A1A", bd=1, 
                              relief="solid", padx=20, pady=20)
        stats_frame.pack(fill='x', pady=10)
        
        tk.Label(stats_frame, text="📊 Forecast Statistics", 
                font=("Times New Roman", 14, "bold"), fg=GOLD, bg="#1A1A1A").pack(anchor='w', pady=(0, 15))
        
        # Create statistics grid
        stats_grid = tk.Frame(stats_frame, bg="#1A1A1A")
        stats_grid.pack(fill='x')
        
        stats_data = [
            ("Total Forecast Sales:", f"₹{total_forecast:,.2f}", GREEN),
            ("Average Daily Sales:", f"₹{avg_daily:,.2f}", GOLD),
            ("Daily Std Deviation:", f"₹{std_daily:,.2f}", BLUE),
            ("Maximum Daily Sales:", f"₹{max_value:,.2f} ({max_day})", GREEN),
            ("Minimum Daily Sales:", f"₹{min_value:,.2f} ({min_day})", ORANGE),
            ("Forecast Range:", f"₹{min_value:,.2f} - ₹{max_value:,.2f}", GOLD),
            ("Confidence Range:", f"₹{np.mean(lower_bound):,.2f} - ₹{np.mean(upper_bound):,.2f}", PURPLE)
        ]
        
        for i, (label, value, color) in enumerate(stats_data):
            row = tk.Frame(stats_grid, bg="#1A1A1A")
            row.pack(fill='x', pady=5)
        
            tk.Label(row, text=label, font=("Times New Roman", 11), 
                    fg="white", bg="#1A1A1A", width=25, anchor='w').pack(side='left')
            tk.Label(row, text=value, font=("Times New Roman", 11, "bold"), 
                    fg=color, bg="#1A1A1A", anchor='w').pack(side='left')
        
        # Add recommendations
        rec_frame = tk.Frame(parent, bg="#1A1A1A", bd=1, 
                            relief="solid", padx=20, pady=20)
        rec_frame.pack(fill='x', pady=10)
        
        tk.Label(rec_frame, text="💡 Business Recommendations", 
                font=("Times New Roman", 14, "bold"), fg=GOLD, bg="#1A1A1A").pack(anchor='w', pady=(0, 15))
        
        recommendations = [
            f"• Peak demand expected on {max_day} - consider increasing stock",
            f"• Average daily sales forecast: ₹{avg_daily:,.2f} - plan inventory accordingly",
            f"• Maintain buffer stock for {min_day} when sales are lower",
            "• Monitor actual vs predicted daily to adjust forecasts",
            "• Consider promotions on low-sales days to boost revenue"
        ]
        
        for rec in recommendations:
            tk.Label(rec_frame, text=rec, font=("Times New Roman", 10), 
                    fg="white", bg="#1A1A1A", anchor='w', justify='left').pack(anchor='w', pady=2)
        
        # Add daily breakdown table
        daily_frame = tk.Frame(parent, bg="#1A1A1A", bd=1, 
                              relief="solid", padx=20, pady=20)
        daily_frame.pack(fill='x', pady=10)
        
        tk.Label(daily_frame, text="📅 Daily Forecast Breakdown", 
                font=("Times New Roman", 14, "bold"), fg=GOLD, bg="#1A1A1A").pack(anchor='w', pady=(0, 15))
        
        # Create a frame for the table with scrollbars
        table_container = tk.Frame(daily_frame, bg="#1A1A1A")
        table_container.pack(fill='both', expand=True)
        
        # Create a canvas for the table
        table_canvas = tk.Canvas(table_container, bg="#1A1A1A", highlightthickness=0, height=200)
        table_vscrollbar = ttk.Scrollbar(table_container, orient="vertical", command=table_canvas.yview)
        table_hscrollbar = ttk.Scrollbar(table_container, orient="horizontal", command=table_canvas.xview)
        
        # Create scrollable frame inside canvas
        table_scrollable_frame = tk.Frame(table_canvas, bg="#1A1A1A")
        
        table_scrollable_frame.bind(
            "<Configure>",
            lambda e: table_canvas.configure(scrollregion=table_canvas.bbox("all"))
        )
        
        table_canvas.create_window((0, 0), window=table_scrollable_frame, anchor="nw")
        table_canvas.configure(yscrollcommand=table_vscrollbar.set, xscrollcommand=table_hscrollbar.set)
        
        # Grid layout for canvas and scrollbars
        table_canvas.grid(row=0, column=0, sticky="nsew")
        table_vscrollbar.grid(row=0, column=1, sticky="ns")
        table_hscrollbar.grid(row=1, column=0, sticky="ew")
        
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)
        
        # Table headers
        headers = ["Day", "Date", "Predicted Sales", "Confidence Range"]
        for col, header in enumerate(headers):
            tk.Label(table_scrollable_frame, text=header, font=("Times New Roman", 10, "bold"), 
                    fg=GOLD, bg="#1A1A1A", width=20).grid(row=0, column=col, padx=5, pady=5, sticky='w')
        
        # Add data rows (show all days)
        for i in range(len(predictions)):
            day_num = i + 1
            date_str = future_dates[i].strftime('%Y-%m-%d')
            pred_val = predictions[i]
            conf_range = f"₹{lower_bound[i]:,.2f} - ₹{upper_bound[i]:,.2f}"
            
            row_data = [f"Day {day_num}", date_str, f"₹{pred_val:,.2f}", conf_range]
            
            for col, data in enumerate(row_data):
                color = "white" if col < 3 else BLUE
                tk.Label(table_scrollable_frame, text=data, font=("Times New Roman", 9), 
                        fg=color, bg="#1A1A1A", width=20).grid(row=i+1, column=col, 
                                                             padx=5, pady=2, sticky='w')

    def plot_simple_forecast(self, predictions, forecast_days, target):
        """Plot simple forecast without time series"""
        
        # Clear previous
        for w in self.chart_area.winfo_children(): 
            w.destroy()
        
        # Create notebook
        plot_notebook = ttk.Notebook(self.chart_area)
        plot_notebook.pack(fill='both', expand=True)
        
        # Tab for forecast
        forecast_tab = tk.Frame(plot_notebook, bg="#151515")
        plot_notebook.add(forecast_tab, text="Future Sales Forecast")
        
        # ================== UPDATED SECTION: FORECAST STATISTICS WITH SCROLLBARS ==================
        # Tab for forecast statistics with scrollbars
        stats_tab = tk.Frame(plot_notebook, bg="#151515")
        plot_notebook.add(stats_tab, text="Forecast Statistics")
        
        # Create a canvas for vertical scrolling
        stats_canvas = tk.Canvas(stats_tab, bg="#151515", highlightthickness=0)
        stats_vscrollbar = ttk.Scrollbar(stats_tab, orient="vertical", command=stats_canvas.yview)
        stats_hscrollbar = ttk.Scrollbar(stats_tab, orient="horizontal", command=stats_canvas.xview)
        
        # Create scrollable frame inside canvas
        stats_scrollable_frame = tk.Frame(stats_canvas, bg="#151515")
        
        stats_scrollable_frame.bind(
            "<Configure>",
            lambda e: stats_canvas.configure(scrollregion=stats_canvas.bbox("all"))
        )
        
        stats_canvas.create_window((0, 0), window=stats_scrollable_frame, anchor="nw")
        stats_canvas.configure(yscrollcommand=stats_vscrollbar.set, xscrollcommand=stats_hscrollbar.set)
        
        # Grid layout for canvas and scrollbars
        stats_canvas.grid(row=0, column=0, sticky="nsew")
        stats_vscrollbar.grid(row=0, column=1, sticky="ns")
        stats_hscrollbar.grid(row=1, column=0, sticky="ew")
        
        stats_tab.grid_rowconfigure(0, weight=1)
        stats_tab.grid_columnconfigure(0, weight=1)
        
        # Bind mousewheel scrolling
        def _on_mousewheel(event):
            stats_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        stats_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Display simple forecast statistics
        self.display_simple_forecast_stats_scrollable(stats_scrollable_frame, predictions, forecast_days, target)
        
        # Create figure for forecast chart
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1A1A1A')
        ax.set_facecolor('#1A1A1A')
        
        # Create day numbers
        days = list(range(1, forecast_days + 1))
        
        # Plot predictions
        ax.plot(days, predictions, 'r-', linewidth=2, marker='o', 
                markersize=5, label='Predicted Sales')
        
        # Fill area under curve
        ax.fill_between(days, predictions, alpha=0.3, color='red')
        
        # Calculate statistics
        avg_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)
        
        # Add average line
        ax.axhline(y=avg_prediction, color='gold', linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'Average: ₹{avg_prediction:,.2f}')
        
        # Add confidence band
        upper_band = [p + std_prediction for p in predictions]
        lower_band = [max(0, p - std_prediction) for p in predictions]
        ax.fill_between(days, lower_band, upper_band, 
                        color='gold', alpha=0.1, label='±1 Std Dev')
        
        # Add peak day annotation
        peak_day = days[np.argmax(predictions)]
        peak_value = np.max(predictions)
        ax.annotate(f'Peak: Day {peak_day}\n₹{peak_value:,.2f}', 
                   xy=(peak_day, peak_value), xytext=(peak_day, peak_value * 1.1),
                   fontsize=10, color='white', ha='center',
                   arrowprops=dict(arrowstyle='->', color='white', alpha=0.7),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="#333", edgecolor="gold"))
        
        # Formatting
        ax.set_xlabel('Day', color='white', fontsize=12)
        ax.set_ylabel(target, color='white', fontsize=12)
        ax.set_title(f'Future Sales Forecast ({forecast_days} Days Ahead)', 
                    color='gold', fontsize=14, pad=20)
        
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1A1A1A', edgecolor='white', 
                 labelcolor='white', fontsize=10)
        ax.grid(True, alpha=0.2, color='gray')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        
        plt.tight_layout()
        
        # Display chart
        canvas = FigureCanvasTkAgg(fig, master=forecast_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def display_simple_forecast_stats_scrollable(self, parent, predictions, forecast_days, target):
        """Display statistics for simple forecast in a scrollable frame"""
        
        # Calculate statistics
        total_forecast = np.sum(predictions)
        avg_daily = np.mean(predictions)
        std_daily = np.std(predictions)
        max_value = np.max(predictions)
        min_value = np.min(predictions)
        max_day = np.argmax(predictions) + 1
        min_day = np.argmin(predictions) + 1
        
        # Display statistics
        stats_frame = tk.Frame(parent, bg="#1A1A1A", bd=1, 
                              relief="solid", padx=20, pady=20)
        stats_frame.pack(fill='x', pady=10)
        
        tk.Label(stats_frame, text="📊 Forecast Statistics", 
                font=("Times New Roman", 14, "bold"), fg=GOLD, bg="#1A1A1A").pack(anchor='w', pady=(0, 15))
        
        # Create statistics grid
        stats_grid = tk.Frame(stats_frame, bg="#1A1A1A")
        stats_grid.pack(fill='x')
        
        stats_data = [
            ("Forecast Period:", f"{forecast_days} days", GOLD),
            ("Total Forecast Sales:", f"₹{total_forecast:,.2f}", GREEN),
            ("Average Daily Sales:", f"₹{avg_daily:,.2f}", GOLD),
            ("Daily Std Deviation:", f"₹{std_daily:,.2f}", BLUE),
            ("Maximum Daily Sales:", f"₹{max_value:,.2f} (Day {max_day})", GREEN),
            ("Minimum Daily Sales:", f"₹{min_value:,.2f} (Day {min_day})", ORANGE),
            ("Forecast Range:", f"₹{min_value:,.2f} - ₹{max_value:,.2f}", GOLD)
        ]
        
        for i, (label, value, color) in enumerate(stats_data):
            row = tk.Frame(stats_grid, bg="#1A1A1A")
            row.pack(fill='x', pady=5)
            
            tk.Label(row, text=label, font=("Times New Roman", 11), 
                    fg="white", bg="#1A1A1A", width=25, anchor='w').pack(side='left')
            tk.Label(row, text=value, font=("Times New Roman", 11, "bold"), 
                    fg=color, bg="#1A1A1A", anchor='w').pack(side='left')
        
        # Add daily breakdown with scrollbars
        daily_frame = tk.Frame(parent, bg="#1A1A1A", bd=1, 
                              relief="solid", padx=20, pady=20)
        daily_frame.pack(fill='x', pady=10)
        
        tk.Label(daily_frame, text="📅 Daily Forecast", 
                font=("Times New Roman", 14, "bold"), fg=GOLD, bg="#1A1A1A").pack(anchor='w', pady=(0, 15))
        
        # Create a frame for the table with scrollbars
        table_container = tk.Frame(daily_frame, bg="#1A1A1A")
        table_container.pack(fill='both', expand=True, height=150)
        
        # Create a canvas for the table
        table_canvas = tk.Canvas(table_container, bg="#1A1A1A", highlightthickness=0)
        table_vscrollbar = ttk.Scrollbar(table_container, orient="vertical", command=table_canvas.yview)
        table_hscrollbar = ttk.Scrollbar(table_container, orient="horizontal", command=table_canvas.xview)
        
        # Create scrollable frame inside canvas
        table_scrollable_frame = tk.Frame(table_canvas, bg="#1A1A1A")
        
        table_scrollable_frame.bind(
            "<Configure>",
            lambda e: table_canvas.configure(scrollregion=table_canvas.bbox("all"))
        )
        
        table_canvas.create_window((0, 0), window=table_scrollable_frame, anchor="nw")
        table_canvas.configure(yscrollcommand=table_vscrollbar.set, xscrollcommand=table_hscrollbar.set)
        
        # Grid layout for canvas and scrollbars
        table_canvas.grid(row=0, column=0, sticky="nsew")
        table_vscrollbar.grid(row=0, column=1, sticky="ns")
        table_hscrollbar.grid(row=1, column=0, sticky="ew")
        
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)
        
        # Table headers
        headers = ["Day", "Predicted Sales", "Trend"]
        for col, header in enumerate(headers):
            tk.Label(table_scrollable_frame, text=header, font=("Times New Roman", 10, "bold"), 
                    fg=GOLD, bg="#1A1A1A", width=15).grid(row=0, column=col, padx=5, pady=5, sticky='w')
        
        # Add data rows (show all days)
        for i in range(len(predictions)):
            day_num = i + 1
            pred_val = predictions[i]
            
            # Determine trend arrow
            if i > 0:
                trend = "↗️" if pred_val > predictions[i-1] else "↘️" if pred_val < predictions[i-1] else "➡️"
            else:
                trend = "➡️"
            
            row_data = [f"Day {day_num}", f"₹{pred_val:,.2f}", trend]
            
            for col, data in enumerate(row_data):
                color = "white" if col < 2 else ("green" if trend == "↗️" else "red" if trend == "↘️" else "gray")
                tk.Label(table_scrollable_frame, text=data, font=("Times New Roman", 9), 
                        fg=color, bg="#1A1A1A", width=15).grid(row=i+1, column=col, 
                                                             padx=5, pady=2, sticky='w')

    def save_best_model(self):
        """Save the best performing model based on metrics"""
        if not self.model_results:
            messagebox.showwarning("Warning", "No trained models to save!")
            return
        
        # Determine best model
        best_model_key = None
        best_score = -float('inf')
        
        for key, results in self.model_results.items():
            # Extract score based on problem type
            score_str = results.get("R2_Acc", "0")
            try:
                score = float(score_str)
                if score > best_score:
                    best_score = score
                    best_model_key = key
            except:
                continue
        
        if not best_model_key:
            messagebox.showwarning("Warning", "Could not determine best model!")
            return
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title=f"Save Best Model: {best_model_key}",
            initialfile=f"best_model_{best_model_key.split(' - ')[0]}.pkl"
        )
        
        if file_path:
            try:
                model_data = {
                    'model': self.model_results[best_model_key]['model'],
                    'model_info': {
                        'name': best_model_key,
                        'algorithm': self.model_results[best_model_key]['Algo'],
                        'target': self.model_results[best_model_key]['Target'],
                        'type': self.model_results[best_model_key]['Type'],
                        'score': best_score,
                        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'feature_names': list(self.current_df.columns.drop([self.model_results[best_model_key]['Target']])) 
                    if self.current_df is not None else None
                }
                
                joblib.dump(model_data, file_path)
                messagebox.showinfo("Success", 
                                  f"Best model saved successfully!\n\n"
                                  f"Model: {best_model_key}\n"
                                  f"Score: {best_score:.4f}\n"
                                  f"Saved to: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model:\n{str(e)}")


    # ============== OUTCOME & ANALYSIS DASHBOARD (Button 6) ==============

    def outcome_premium_button(self, parent, text, command, width=20, height=1, bg="#D4AF37"):
        """Outcome-Specific high-fidelity button styling"""
        btn = tk.Button(parent, text=text, command=command, width=width, height=height,
                        font=("Times New Roman", 10, "bold"), bg="#1A1A1A", fg=bg,
                        activebackground=bg, activeforeground="#1A1A1A",
                        bd=1, relief="solid", highlightthickness=0, cursor="hand2")
        btn.bind("<Enter>", lambda e: btn.config(bg=bg, fg="#1A1A1A"))
        btn.bind("<Leave>", lambda e: btn.config(bg="#1A1A1A", fg=bg))
        return btn

    def outcome_input_dialog(self, title, prompt, callback):
        """Outcome-Specific modern-styled input dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("350x180")
        dialog.configure(bg="#050505")
        dialog.transient(self.root)
        dialog.grab_set()
        tk.Label(dialog, text=title, font=("Times New Roman", 14, "bold"), fg="#D4AF37", bg="#050505").pack(pady=15)
        tk.Label(dialog, text=prompt, font=("Times New Roman", 10), fg="white", bg="#050505").pack(pady=5)
        entry = tk.Entry(dialog, font=("Times New Roman", 11), bg="#111", fg="white", insertbackground="white", bd=1, relief="solid")
        entry.pack(pady=10, padx=40, fill="x")
        entry.focus_set()
        def on_submit():
            val = entry.get()
            dialog.destroy()
            if val: callback(val)
        self.outcome_premium_button(dialog, "SUBMIT", on_submit, width=15).pack(pady=10)
        dialog.bind("<Return>", lambda e: on_submit())

    def trigger_smart_forecast(self):
        """Intelligent trigger for forecasting from the Outcome Page"""
        if not self.model_results:
            messagebox.showwarning("Warning", "No trained models found! Please train a model first.")
            return
        best_model = None
        for key in self.model_results:
            if self.model_results[key].get("Type") == "Regression":
                best_model = key
                break
        if not best_model:
            messagebox.showwarning("Warning", "Forecasting requires a trained Regression model.")
            return
        if not hasattr(self, 'res_combo_var'): self.res_combo_var = tk.StringVar(value=best_model)
        else: self.res_combo_var.set(best_model)
        if not hasattr(self, 'forecast_days'): self.forecast_days = tk.StringVar(value="30")
        if not hasattr(self, 'confidence_level'): self.confidence_level = tk.StringVar(value="95")
        self.outcome_input_dialog("Sales Forecasting", "Enter forecast days (1-365):", 
                               lambda days: self._run_final_forecast(days, best_model))

    def _run_final_forecast(self, days, model_key):
        """Internal helper to execute the forecast after input"""
        try:
            d = int(days)
            if d < 1 or d > 365: raise ValueError()
            self.forecast_days.set(str(d))
            self.res_combo_var.set(model_key)
            self.generate_future_predictions()
        except:
            messagebox.showerror("Error", "Please enter a valid number of days (1-365)")

    def outcome_analysis(self, mode="inventory"):
        """
        Unified Strategic Management Suite - HIGH FIDELITY VERSION.
        Replicates the generated design image with icons, glow effects, and glassmorphism styling.
        Real data-driven logic for Inventory and Pricing.
        """
        if self.current_df is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        # Check if we have models (Strict Enforce as per user request)
        if not hasattr(self, 'model_results') or not self.model_results:
             messagebox.showwarning("Training Required", 
                                  "The Outcome Analysis page requires trained ML models.\n"
                                  "Please complete 'Model Training' first.")
             return

        self.current_page = "outcome_analysis"
        self.clear()

        # --- 1. DATA ANALYSIS & FUNCTIONAL LOGIC ---
        try:
            # Prioritize unscaled data for business analysis (fixes negative values from standardization)
            if self.cleaned_df is not None:
                source_df = self.cleaned_df.copy()
            elif self.original_df is not None:
                source_df = self.original_df.copy()
            else:
                source_df = self.current_df.copy()
            
            # --- ROBUST SCORING MAPPING ---
            cols = [c.lower() for c in source_df.columns]
            col_map = {c.lower(): c for c in source_df.columns} # Low -> Original
            
            # 1. Product Name Scoring
            i_scores = []
            for c in source_df.columns:
                cl = c.lower()
                s = 0
                # Heavy penalty for non-product columns
                if 'pid' in cl or 'code' in cl or 'id' in cl or 'price' in cl or 'qty' in cl or 'amount' in cl or 'date' in cl: s = -100
                elif 'cat' in cl or 'grp' in cl or 'group' in cl or 'type' in cl: s = -20
                elif 'brand' in cl: s = -20 # Brand is NOT Product (unless specified like 'Product Brand')
                
                else:
                    if 'product' in cl: s += 50
                    if 'item' in cl: s += 50
                    if 'name' in cl: s += 20
                    if 'desc' in cl: s += 30
                    if 'model' in cl: s += 20
                
                if s > 0: i_scores.append((c, s))
            
            i_scores.sort(key=lambda x: x[1], reverse=True)
            i_col = i_scores[0][0] if i_scores else None
            
            # Fallback 1: Just 'Name' if strict filtered it out
            if not i_col:
                 fallback = next((c for c in source_df.columns if 'name' in c.lower() and 'brand' not in c.lower() and 'cat' not in c.lower()), None)
                 if fallback: i_col = fallback
            
            # Fallback 2: Any 'Item' or 'Product' even if it has 'id' (last resort)
            if not i_col:
                 fallback = next((c for c in source_df.columns if 'item' in c.lower() or 'product' in c.lower()), None)
                 if fallback: i_col = fallback

            if not i_col: i_col = "Item"

            # 2. Category Scoring
            c_scores = []
            for c in source_df.columns:
                if c == i_col: continue # Exclusive
                cl = c.lower()
                s = 0
                if 'id' in cl or 'code' in cl or 'date' in cl or 'price' in cl or 'qty' in cl: s = -100
                
                if 'cat' in cl: s += 50
                if 'class' in cl: s += 40
                if 'grp' in cl or 'group' in cl: s += 30
                if 'type' in cl: s += 30
                if 'brand' in cl: s += 20 # Brand is an acceptable fallback category
                
                if s > 0: c_scores.append((c, s))
                
            c_scores.sort(key=lambda x: x[1], reverse=True)
            cat_col = c_scores[0][0] if c_scores else None

            # --- DECODE COLUMNS IF ENCODED ---
            # If the user has encoded data, these columns will be numeric. We need to decode them back to strings.
            for col_name in [i_col, cat_col]:
                if col_name and col_name in source_df.columns:
                    # Check if numeric
                    if pd.api.types.is_numeric_dtype(source_df[col_name]):
                        # Method 1: Try app's helper
                        decoded = self.get_decoded_column(col_name)
                        # Helper returns same series if fail, or decoded series if success
                        # Check if it actually changed something or returned a valid string series
                        if decoded is not None and not pd.api.types.is_numeric_dtype(decoded):
                             # Align indices just in case
                             if len(decoded) == len(source_df):
                                 source_df[col_name] = decoded.values
                        
                        # Method 2: Fallback to Mapping Dict
                        if pd.api.types.is_numeric_dtype(source_df[col_name]) and hasattr(self, 'brand_encoding_mapping'):
                            if col_name in self.brand_encoding_mapping:
                                mapping = self.brand_encoding_mapping[col_name]
                                # Mapping is Name -> Int. We need Int -> Name
                                rev_map = {v: k for k, v in mapping.items()}
                                source_df[col_name] = source_df[col_name].map(rev_map).fillna(source_df[col_name])
            
            # --- CONTINUE MAPPING ---
            cols = [c.lower() for c in source_df.columns] # Re-scan in case names changed (unlikely)

            # 3. Price & Quantity & Date
            p_col = next((col_map[c] for c in cols if 'price' in c or 'amt' in c or 'unit' in c), None)
            q_col = next((col_map[c] for c in cols if 'qty' in c or 'quantity' in c or 'vol' in c), None)
            date_col = next((col_map[c] for c in cols if 'date' in c or 'time' in c), None)

            # Data Cleaning for Analysis
            if p_col: source_df[p_col] = pd.to_numeric(source_df[p_col], errors='coerce').fillna(10.0)
            if q_col: source_df[q_col] = pd.to_numeric(source_df[q_col], errors='coerce').fillna(1)
            
            # Persist these robustly detected columns for other views (Charts)
            self.last_analysis_cols = {
                'i_col': i_col, 'cat_col': cat_col, 'p_col': p_col, 'q_col': q_col
            }
            
            # Reuse calculated TotalSale if available, else calc
            if 'TotalSale' not in source_df.columns and p_col and q_col:
                 source_df['TotalSale'] = source_df[p_col] * source_df[q_col]
            
            # KPI Calculations
            self.total_revenue = source_df['TotalSale'].sum() if 'TotalSale' in source_df.columns else 0
            self.total_transactions = len(source_df)
            if cat_col and q_col:
                cat_sales = source_df.groupby(cat_col)[q_col].sum()
                if not cat_sales.empty:
                    self.leading_brand = cat_sales.idxmax()
                else:
                    self.leading_brand = "N/A"
            else:
                self.leading_brand = "N/A"
            self.optimal_price_point = source_df[p_col].mean() if p_col and not source_df.empty else 0.0

            # --- CALCULATE PER-ITEM METRICS ---
            
            # Convert Date
            if date_col:
                source_df[date_col] = pd.to_datetime(source_df[date_col], errors='coerce')
                # Global Max Date (Today)
                global_max_date = source_df[date_col].max()
            else:
                global_max_date = pd.Timestamp.now()

            # Group by Product
            agg_dict = {}
            if q_col: agg_dict[q_col] = 'sum'
            if p_col: agg_dict[p_col] = 'mean'
            if date_col: agg_dict[date_col] = ['min', 'max'] 

            item_stats = source_df.groupby(i_col).agg(agg_dict).reset_index()
            
            # Flatten columns
            new_cols = ['product']
            if q_col: new_cols.append('total_qty')
            if p_col: new_cols.append('avg_price')
            if date_col: new_cols.extend(['first_sale', 'last_sale'])
            item_stats.columns = new_cols
            
            # Handle missing columns if any
            if 'total_qty' not in item_stats.columns: item_stats['total_qty'] = 0
            if 'avg_price' not in item_stats.columns: item_stats['avg_price'] = 0

            # Calculate Velocity (Per Item)
            def calc_velocity(row):
                if date_col and 'first_sale' in row:
                    # Days Active = (Last Sale in Dataset OR Now) - First Sale
                    # tailored to avoid penalizing new items
                    days = (global_max_date - row['first_sale']).days
                    days = max(days, 1) # Minimum 1 day
                    return row['total_qty'] / days
                else:
                    return row['total_qty'] / 30 # Default to monthly avg assumption

            item_stats['velocity'] = item_stats.apply(calc_velocity, axis=1)
            
            # Global Averages for Pricing Matrix
            avg_velocity_global = item_stats['velocity'].mean()
            avg_price_global = item_stats['avg_price'].mean()
            
            # Category Map
            cat_map = {}
            if cat_col:
                try:
                    # Safety check: ensure both columns exist
                    if i_col in source_df.columns and cat_col in source_df.columns:
                        cat_map = source_df.drop_duplicates(subset=[i_col]).set_index(i_col)[cat_col].fillna("General").to_dict()
                except:
                    print("Category mapping failed, defaulting to General")
                    pass

            # --- AGGREGATE INSIGHTS FOR "AI INSIGHTS" ---
            self.latest_insights = {
                "fast_movers": 0, "slow_movers": 0, "dead_stock": 0,
                "cash_cows": 0, "stars": 0, "dogs": 0,
                "potential_rev_gain": 0.0,
                "action_items": []
            }

            for _, row in item_stats.iterrows():
                # Re-run logic briefly for aggregation
                vel = row['velocity']
                price = row['avg_price']
                
                is_fast = vel > avg_velocity_global * 1.5
                is_slow = vel < avg_velocity_global * 0.5
                if is_fast: self.latest_insights["fast_movers"] += 1
                if is_slow: self.latest_insights["slow_movers"] += 1
                if vel == 0: self.latest_insights["dead_stock"] += 1

                is_prem = price > avg_price_global
                is_vol = vel > avg_velocity_global
                
                if is_vol and not is_prem: # Cash Cow
                    self.latest_insights["cash_cows"] += 1
                    # Potential gain: 5% price increase * volume
                    self.latest_insights["potential_rev_gain"] += (price * 0.05) * row['total_qty']
                elif is_vol and is_prem: # Star
                    self.latest_insights["stars"] += 1
                elif not is_vol and not is_prem: # Dog
                    self.latest_insights["dogs"] += 1
            
            # Sort top 3 actionable items
            top_opps = item_stats.sort_values('velocity', ascending=False).head(3)
            for _, row in top_opps.iterrows():
                self.latest_insights["action_items"].append(f"• {row['product']}: Primary volume driver. Ensure 100% availability.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Analysis failed: {e}")
            return

        # --- 2. THEME & UI ---
        BG_D = "#050505"; GOLD = "#D4AF37"; ACCENT = "#1A1A1A"; TEXT = "#E0E0E0"
        self.root.configure(bg=BG_D)

        # Header Section
        header = tk.Frame(self.root, bg=BG_D, height=140)
        header.pack(fill="x")
        header.pack_propagate(False)

        # Exit Button
        tk.Button(header, text="← EXIT", font=("Times New Roman", 9, "bold"), bg=ACCENT, fg=TEXT, 
                activebackground=GOLD, bd=0, padx=15, pady=5, cursor="hand2", command=self.feature_page).place(x=30, y=25)

        # Title
        title_text = "Analysis Results"
        if mode == "inventory": title_text = "Inventory Planning Strategy"
        elif mode == "pricing": title_text = "Pricing Optimization Strategy"
        
        title = tk.Label(header, text=title_text, 
                        font=("Times New Roman", 28, "bold"), fg=GOLD, bg=BG_D)
        title.place(relx=0.5, y=50, anchor="n")

        # Metadata Bar
        meta_bar = tk.Frame(header, bg="#111", height=32, bd=1, relief="solid", highlightbackground="#333", highlightthickness=1)
        meta_bar.pack(side="bottom", fill="x", padx=150, pady=(0, 15))
        meta_bar.pack_propagate(False)
        
        tk.Label(meta_bar, text=f"Data Period: Dynamic | Active Items: {len(item_stats)} | Total Revenue: ₹{self.total_revenue:,.0f}", 
                 font=("Times New Roman", 9), fg="#888", bg="#111").pack(side="left", padx=20)
        
        # Small Chart Option near Table
        tk.Button(meta_bar, text="📊 Show Charts", command=lambda: self.show_outcome_charts(mode), 
                  font=("Times New Roman", 9, "bold"), bg="#1A1A1A", fg=GOLD, bd=0, cursor="hand2").pack(side="right", padx=20, pady=2)

        # --- 4. FOOTER BUTTONS ---
        footer = tk.Frame(self.root, bg=BG_D, pady=30)
        footer.pack(side="bottom", fill="x", padx=40)
        
        self.outcome_premium_button(footer, "INVENTORY PLAN", lambda: self.outcome_analysis(mode="inventory"), 24, 2).pack(side="left", expand=True, padx=5)
        self.outcome_premium_button(footer, "PRICING STRATEGY", lambda: self.outcome_analysis(mode="pricing"), 24, 2).pack(side="left", expand=True, padx=5)
        self.outcome_premium_button(footer, "SALES FORECAST", self.trigger_smart_forecast, 24, 2).pack(side="left", expand=True, padx=5)
        # NEW BUTTON
        self.outcome_premium_button(footer, "Insights", self.show_ai_insights, 24, 2, bg=GOLD).pack(side="left", expand=True, padx=5)
        # Charts now integrated into the view options
        self.outcome_premium_button(footer, "EXPORT REPORT", self.export_strategic_report, 24, 2).pack(side="left", expand=True, padx=5)

        # Charts removed from here (embedded view failed). 
        # Added button in header instead.

        # Main Table Container
        container = tk.Frame(self.root, bg="#0A0A0A", bd=1, relief="solid", highlightbackground="#2A2A2A", highlightthickness=2)
        container.pack(side="top", fill="both", expand=True, padx=50, pady=5)

        # Styled Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Premium.Treeview", background="#0A0A0A", foreground="white", fieldbackground="#0A0A0A", 
                        rowheight=42, font=("Times New Roman", 10), borderwidth=0)
        style.map("Premium.Treeview", background=[('selected', GOLD)], foreground=[('selected', 'black')])
        style.configure("Premium.Treeview.Heading", background="#151515", foreground=GOLD, 
                        font=("Times New Roman", 11, "bold"), borderwidth=0)

        cols = ("P", "C", "D", "CS", "RS", "A", "AP", "SP", "L")
        tree = ttk.Treeview(container, columns=cols, show="headings", style="Premium.Treeview")
        
        heads = ["Product Name", "Category", "Sales/Day", "Velocity Status", "Reorder Pt", "Action", "Cur. Price", "Sug. Price", "Strategy Logic"]
        # Increased last column width to 450 to prevent cut-off
        widths = [220, 140, 90, 130, 90, 130, 90, 90, 450]
        
        for c, h, w in zip(cols, heads, widths):
            tree.heading(c, text=h)
            tree.column(c, width=w, anchor="center" if c not in ["P", "L"] else "w")

        # Grid configuration for scrollbars
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        tree.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbars
        # Scrollbars (Using standard tk.Scrollbar for custom colors)
        vsb = tk.Scrollbar(container, orient="vertical", command=tree.yview, bg="black", troughcolor="black", activebackground="gold", highlightbackground="black", relief="flat")
        vsb.grid(row=0, column=1, sticky="ns")
        
        hsb = tk.Scrollbar(container, orient="horizontal", command=tree.xview, bg="black", troughcolor="black", activebackground="gold", highlightbackground="black", relief="flat")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # --- 3. DATA POPULATION (REAL LOGIC - ROBUST) ---
        try:
            perish_keywords = ["milk", "bread", "dairy", "meat", "fruit", "egg", "fresh", "yogurt", "cheese", "butter"]
            
            # Calculate Robust Thresholds (Quantiles)
            # Avoid skewed means by using medians and percentiles
            vel_median = item_stats['velocity'].median()
            vel_p75 = item_stats['velocity'].quantile(0.75) # Top 25% = Fast
            vel_p25 = item_stats['velocity'].quantile(0.25) # Bottom 25% = Slow
            
            price_median = item_stats['avg_price'].median()
            
            # Handle case where all velocities are same (e.g. 0)
            if vel_p75 == 0: vel_p75 = 0.1
            
            # Sort by velocity for better view
            sorted_stats = item_stats.sort_values('velocity', ascending=False)
            
            for _, row in sorted_stats.iterrows():
                p_name = str(row['product'])
                cat = cat_map.get(row['product'], "General")
                price = row['avg_price']
                velocity = row['velocity']
                
                # --- STOCK LOGIC ---
                # Reorder Point = Velocity * Lead Time (14 Days)
                lead_time = 14
                reorder_point = int(velocity * lead_time)
                
                is_fast_mover = velocity >= vel_p75
                is_slow_mover = velocity <= vel_p25
                
                if is_fast_mover:
                    demand_text = f"High ({velocity:.1f})"
                    stock_status_text = "Velocity: Top 25%"
                    rec_stock_text = f"> {reorder_point}"
                    action_text = "⚡ KEEP STOCKED"
                    inv_tag = "green" 
                elif is_slow_mover:
                    demand_text = f"Low ({velocity:.1f})"
                    stock_status_text = "Velocity: Bot 25%"
                    rec_stock_text = f"< {reorder_point}"
                    action_text = "📉 REDUCE"
                    inv_tag = "red" 
                else:
                    demand_text = f"Med ({velocity:.1f})"
                    stock_status_text = "Velocity: Stable"
                    rec_stock_text = f"~ {reorder_point}"
                    action_text = "✅ MAINTAIN"
                    inv_tag = "white"

                # --- PRICING LOGIC (BCG Matrix style - Percentile Based) ---
                
                is_premium = price > price_median
                is_volume_high = velocity > vel_median
                
                sug_price = price
                pricing_logic = ""
                pricing_tag = ""
                
                if is_volume_high and is_premium:
                    # STAR: Above Median Vol, Above Median Price
                    pricing_logic = f"Top Performer: High Demand & High Value. Priority: Protect Stock."
                    pricing_tag = "green"
                elif is_volume_high and not is_premium:
                    # CASH COW: Above Median Vol, Below Median Price
                    sug_price = price * 1.05
                    pricing_logic = f"High Volume: Strong Demand. Strategy: Test +5% Price Increase."
                    pricing_tag = "blue" 
                elif not is_volume_high and is_premium:
                    # PROBLEM CHILD: Below Median Vol, Above Median Price
                    sug_price = price * 0.90
                    pricing_logic = "Opportunity: High Value but Low Volume. Strategy: Discount 10%."
                    pricing_tag = "orange"
                else:
                    # DOG: Below Median Vol, Below Median Price
                    sug_price = price * 0.80
                    pricing_logic = "Low Performance: Low Demand. Strategy: Clearance / Liquidation."
                    pricing_tag = "red"
                    
                # --- FILTERING PROPERLY BASED ON BUTTON CLICK ---
                
                display_tag = "white" # Default text color
                
                if mode == "inventory":
                    display_tag = inv_tag
                    logic_to_show = f"{action_text}. Reorder when stock hits {reorder_point} units."
                    sug_p_show = "-"
                elif mode == "pricing":
                    display_tag = pricing_tag
                    logic_to_show = pricing_logic
                    sug_p_show = f"₹{sug_price:.2f}"
                else:
                    # ALL Mode: Priority to specific alerts
                    if pricing_tag == "red": display_tag = "red" # Dogs are bad
                    elif inv_tag == "red": display_tag = "red" # Dead stock is bad
                    elif pricing_tag == "green": display_tag = "green" # Stars are good
                    else: display_tag = "white"
                    
                    # Fix for UnboundLocalError
                    logic_to_show = pricing_logic  # Default to pricing logic in ALL view
                    sug_p_show = f"₹{sug_price:.2f}"
                
                tree.insert("", "end", values=(
                    p_name[:30], str(cat)[:15], demand_text, stock_status_text, rec_stock_text, action_text,
                    f"₹{price:.2f}", sug_p_show, logic_to_show
                ), tags=(display_tag,))

        except Exception as e:
            messagebox.showerror("Error", f"Table Logic Error: {e}")

        tree.tag_configure("red", foreground="#FF5252") # Soft Vibrant Red
        tree.tag_configure("green", foreground="#2ECC71") # Emerald Green
        tree.tag_configure("blue", foreground="#3498DB") # Bright Blue
        tree.tag_configure("orange", foreground="#FFA726") # Muted Orange
        tree.tag_configure("gray", foreground="#95A5A6") # Silver Gray
        tree.tag_configure("white", foreground="white")

        # --- Footer moved to top of layout construction ---
        # (Buttons are already packed)

    
    def show_outcome_charts(self, mode="all"):
        """Display strategic charts for Outcome Analysis"""
        # We need the calculated stats from outcome_analysis. 
        # Since we don't return them, we must re-calculate or rely on self.latest_insights and self.current_df
        # Re-running lightweight aggregation is safer.
        if self.current_df is None: return

        try:
             # Select Data Source (Unscaled)
             if self.cleaned_df is not None:
                 df = self.cleaned_df.copy()
             elif self.original_df is not None:
                 df = self.original_df.copy()
             else:
                 df = self.current_df.copy()
             
             # Attempt to use robustly detected columns from the main analysis
             p_col = None; q_col = None
             if hasattr(self, 'last_analysis_cols'):
                 p_col = self.last_analysis_cols.get('p_col')
                 q_col = self.last_analysis_cols.get('q_col')
            
             # Fallback if not ready
             if not (p_col and q_col):
                cols = [c.lower() for c in df.columns]
                col_map = {c.lower(): c for c in df.columns}
                p_col = next((col_map[c] for c in cols if 'price' in c or 'amt' in c), None)
                q_col = next((col_map[c] for c in cols if 'qty' in c or 'vol' in c), None)
             
             if not (p_col and q_col):
                 messagebox.showinfo("Info", "Need Price/Qty columns for charts.")
                 return
             
             # ENSURE NUMERIC (Fix for "can only concatenate str" and plotting errors)
             # The raw/cleaned df might still have these as strings
             df[p_col] = pd.to_numeric(df[p_col], errors='coerce').fillna(0)
             df[q_col] = pd.to_numeric(df[q_col], errors='coerce').fillna(0)
             
             # Aggregations
             # Product Agg varies... let's just use what we have in current_df
             
             # Create Modal
             modal = tk.Toplevel(self.root)
             modal.title("📊 Strategic Visual Analysis")
             modal.geometry("1100x800")
             modal.configure(bg="#050505")
             
             notebook = ttk.Notebook(modal)
             notebook.pack(fill='both', expand=True, padx=10, pady=10)
             
             # 1. BCG Matrix (Price vs Volume) - Show for Pricing or All
             if mode == "pricing" or mode == "all":
                 tab1 = tk.Frame(notebook, bg="black")
                 notebook.add(tab1, text="BCG Matrix (Price vs Vol)")
                 
                 fig1, ax1 = plt.subplots(figsize=(8, 6))
                 # Scatter plot
                 subset = df.head(500) # Limit for speed
                 sns.scatterplot(x=subset[p_col], y=subset[q_col], ax=ax1, color="#D4AF37", alpha=0.6)
                 ax1.set_title("Price vs Volume Strategy Map", color="white")
                 ax1.set_xlabel("Unit Price", color="white")
                 ax1.set_ylabel("Sales Volume", color="white")
                 ax1.tick_params(colors="white")
                 ax1.spines['bottom'].set_color('white')
                 ax1.spines['left'].set_color('white')
                 ax1.set_facecolor("#111")
                 fig1.patch.set_facecolor("#111")
                 
                 # Zones annotations
                 avg_p = df[p_col].mean()
                 avg_q = df[q_col].mean()
                 ax1.axvline(avg_p, color='gray', linestyle='--')
                 ax1.axhline(avg_q, color='gray', linestyle='--')
                 ax1.text(df[p_col].max()*0.9, df[q_col].max()*0.9, "STARS", color="#2ECC71", fontsize=12, fontweight='bold')
                 ax1.text(df[p_col].min()*1.1, df[q_col].min()*1.1, "DOGS", color="#FF5252", fontsize=12, fontweight='bold')
    
                 canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
                 canvas1.draw()
                 canvas1.get_tk_widget().pack(fill="both", expand=True)
             
             # 2. Velocity Distribution - Show for Inventory or All
             if mode == "inventory" or mode == "all":
                 tab2 = tk.Frame(notebook, bg="black")
                 notebook.add(tab2, text="Sales Velocity Distribution")
                 
                 fig2, ax2 = plt.subplots(figsize=(8, 6))
                 sns.histplot(df[q_col], bins=30, kde=True, ax=ax2, color="#3498DB")
                 ax2.set_title("Sales Frequency Distribution", color="white")
                 ax2.tick_params(colors="white"); ax2.xaxis.label.set_color('white'); ax2.yaxis.label.set_color('white')
                 ax2.set_facecolor("#111"); fig2.patch.set_facecolor("#111")
                 
                 canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
                 canvas2.draw()
                 canvas2.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Could not generate charts: {e}")

    def show_ai_insights(self):
        """Display System Outcome & Executive Summary Dashboard"""
        
        # --- THEME COLORS ---
        BG_MAIN = "#1E1E1E" # Dark Gray/Black like screenshot
        BG_CARD = "#2D2D30" # Lighter Gray for cards
        # TEXT_GOLD = "#D4AF37"
        TEXT_GOLD = "#FFC107" # Brighter Amber/Gold
        TEXT_GREEN = "#4CAF50" # Material Green
        TEXT_WHITE = "#FFFFFF"
        TEXT_SUB = "#AAAAAA"

        modal = tk.Toplevel(self.root)
        modal.title("System Outcome & Executive Summary")
        modal.geometry("1100x700")
        modal.configure(bg=BG_MAIN)
        
        # --- HEADER ---
        header_frame = tk.Frame(modal, bg=BG_MAIN, pady=20)
        header_frame.pack(fill="x")
        
        # Back Button (Top Left)
        tk.Button(header_frame, text="⬅ BACK", command=modal.destroy,
                 font=("Times New Roman", 10, "bold"), bg=TEXT_GOLD, fg=BG_MAIN,
                 relief="flat", bd=0, padx=10, pady=5).place(x=20, y=20)
        
        tk.Label(header_frame, text="SYSTEM OUTCOME & EXECUTIVE SUMMARY", 
                 font=("Times New Roman", 22, "bold"), fg=TEXT_GOLD, bg=BG_MAIN).pack()

        # --- KPI ROW (Business Performance) ---
        kpi_container = tk.Frame(modal, bg=BG_MAIN, pady=10)
        kpi_container.pack(fill="x", padx=40)
        
        # Label: Business Performance
        tk.Label(kpi_container, text="📊 Business Performance (From Dataset)", 
                 font=("Times New Roman", 14, "bold"), fg=TEXT_GOLD, bg=BG_MAIN, anchor="w").pack(fill="x", pady=(0, 10))
        
        grid_frame = tk.Frame(kpi_container, bg=BG_MAIN)
        grid_frame.pack(fill="x")

        # Helper to create Card
        def create_card(parent, title, value, subtext):
            card = tk.Frame(parent, bg=BG_CARD, padx=15, pady=15, width=240, height=140)
            card.pack_propagate(False) # Fixed size
            
            val_lbl = tk.Label(card, text=value, font=("Times New Roman", 20, "bold"), fg=TEXT_GOLD if "N/A" not in value else TEXT_WHITE, bg=BG_CARD)
            val_lbl.pack(expand=True)
            
            title_lbl = tk.Label(card, text=title, font=("Times New Roman", 10), fg=TEXT_WHITE, bg=BG_CARD)
            title_lbl.pack(side="bottom", pady=2)
            
            sub_lbl = tk.Label(card, text=subtext, font=("Times New Roman", 8), fg=TEXT_SUB, bg=BG_CARD)
            sub_lbl.pack(side="bottom")
            
            return card

        # Card 1: Revenue
        rev_val = f"₹{self.total_revenue:,.2f}" if hasattr(self, 'total_revenue') else "₹0.00"
        c1 = create_card(grid_frame, "Total Revenue Generated", rev_val, "Overall Sales")
        c1.pack(side="left", padx=5, fill="both", expand=True)

        # Card 2: Leading Brand
        brand_val = str(self.leading_brand) if hasattr(self, 'leading_brand') else "N/A"
        c2 = create_card(grid_frame, "Market Leading Brand", brand_val, "Top Category")
        c2.pack(side="left", padx=5, fill="both", expand=True)

        # Card 3: Transactions
        tx_val = f"{self.total_transactions:,}" if hasattr(self, 'total_transactions') else "0"
        c3 = create_card(grid_frame, "Transactions Validated", tx_val, "Total Volume")
        c3.pack(side="left", padx=5, fill="both", expand=True)

        # Card 4: Model Performance
        # Try to find model stats
        model_score = "N/A"
        model_name = "Not Trained"
        
        if hasattr(self, 'model_results') and self.model_results:
            best_s = -float('inf')
            best_n = "Unknown"
            
            for key, val in self.model_results.items():
                try:
                    # Score is stored as string "0.9523" in 'R2_Acc'
                    s = float(val.get('R2_Acc', 0))
                    if s > best_s:
                        best_s = s
                        best_n = key
                except:
                    continue
            
            if best_s > -float('inf'):
                model_score = f"Score: {best_s:.4f}"
                # Extract just the algo name if possible (remove target)
                if " - " in best_n:
                    model_name = best_n.split(" - ")[-1]
                else:
                    model_name = best_n
            
        c4 = tk.Frame(grid_frame, bg=BG_CARD, padx=15, pady=15, width=240, height=140)
        c4.pack_propagate(False)
        
        # Display Name BIG, Score Small
        tk.Label(c4, text=str(model_name)[:20], font=("Times New Roman", 15, "bold"), fg=TEXT_GREEN, bg=BG_CARD, wraplength=220).pack(expand=True)
        tk.Label(c4, text=model_score, font=("Times New Roman", 12, "bold"), fg=TEXT_WHITE, bg=BG_CARD).pack(side="bottom", pady=5)
        tk.Label(c4, text="Best Performing Model", font=("Times New Roman", 9), fg=TEXT_SUB, bg=BG_CARD).pack(side="bottom")
        c4.pack(side="left", padx=5, fill="both", expand=True)

        # --- CHECKLIST SECTION ---
        check_container = tk.Frame(modal, bg=BG_MAIN, pady=40)
        check_container.pack(fill="x", padx=40)
        
        tk.Label(check_container, text="☑ System Ready for Deployment", 
                 font=("Times New Roman", 16, "bold"), fg=TEXT_GREEN, bg=BG_MAIN, anchor="w").pack(fill="x", pady=(0, 15))
        
        # Check Item Helper
        def add_check(text, condition=True):
            icon = "✅" if condition else "⏳"
            color = TEXT_WHITE if condition else TEXT_SUB
            row = tk.Frame(check_container, bg=BG_MAIN)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=icon, font=("Times New Roman", 12), bg=BG_MAIN, fg=TEXT_WHITE).pack(side="left", padx=(10, 10))
            tk.Label(row, text=text, font=("Times New Roman", 11), bg=BG_MAIN, fg=color).pack(side="left")

        # Logic Checks
        has_data = hasattr(self, 'current_df') and self.current_df is not None
        has_eda = hasattr(self, 'item_stats') or hasattr(self, 'total_revenue')
        
        # FIX: Check model_results instead of legacy 'model' var
        has_model = hasattr(self, 'model_results') and len(self.model_results) > 0
        
        # Data Loading is complete if we have data
        add_check("1. Data Loading", condition=has_data)
        add_check("2. Data Preprocessing", condition=has_data)
        add_check("3. EDA Analysis", condition=has_eda)
        add_check("4. ML Model Training", condition=has_model)
        add_check("5. Results Analysis", condition=has_model)
        add_check("6. Outcome & Planning (Inventory, Pricing, Forecasting)", condition=True) # We are here
        
        if has_model:
            status_text = "ML Models: Trained successfully"
        else:
            status_text = "ML Models: Pending Training"
            
        tk.Label(check_container, text=status_text, font=("Times New Roman", 10, "bold"), fg=TEXT_SUB, bg=BG_MAIN, anchor="w").pack(fill="x", pady=(20, 5), padx=40)
        tk.Label(check_container, text="Dataset Analysis Complete.", font=("Times New Roman", 10, "italic"), fg=TEXT_SUB, bg=BG_MAIN, anchor="w").pack(fill="x", padx=40)

        # Close
        tk.Button(modal, text="CLOSE", command=modal.destroy, font=("Times New Roman", 10, "bold"), bg="#333", fg="white", bd=0, padx=20, pady=10).pack(pady=10)

    def export_strategic_report(self):
        """Generates a professional PDF report using real data analysis tags"""
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf", 
                                                    filetypes=[("PDF files", "*.pdf")],
                                                    title="Save Strategic Executive Report")
            if not file_path: return
            
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Header
            title_style = styles['Title']
            title_style.textColor = colors.HexColor("#D4AF37")
            elements.append(Paragraph("EXECUTIVE STRATEGIC SUMMARY", title_style))
            elements.append(Spacer(1, 20))
            
            # Stats Table
            data = [
                ["Metric", "Value", "Business Insight"],
                ["Total Revenue", f"₹{self.total_revenue:,.2f}", "Overall market capture in current period."],
                ["Total Volume", f"{self.total_transactions:,} Units", "Transaction velocity across all categories."],
                ["Leading Category", f"{self.leading_brand}", "Core revenue driver for current inventory."],
            ]
            
            if hasattr(self, 'optimal_price_point'):
                 data.append(["Optimal Price Point", f"₹{self.optimal_price_point:.2f}", "Maximum yield equilibrium identified."])
            t = Table(data, colWidths=[120, 100, 280])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D4AF37")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0,0), (-1,-1), 1, colors.grey),
                ('BOTTOMPADDING', (0,0), (-1,0), 10),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 30))
            
            # Forecast Table
            if hasattr(self, 'forecast_dates') and hasattr(self, 'forecast_values'):
                elements.append(Paragraph("Quarterly Performance Projections", styles['Heading2']))
                f_data = [["Period", "Projected Revenue Growth"]]
                for d, v in zip(self.forecast_dates, self.forecast_values):
                    f_data.append([d, f"₹{v:,.2f}"])
                
                ft = Table(f_data, colWidths=[200, 300])
                ft.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ]))
                elements.append(ft)
            else:
                elements.append(Paragraph("Forecast data not available. Please run predictive modeling first.", styles['Italic']))
            
            doc.build(elements)
            messagebox.showinfo("Success", "Strategic Report exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Report Export Failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SalesApp(root)
    root.mainloop()
