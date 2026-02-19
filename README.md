# Supermarket Sales Analysis System

A comprehensive desktop application for analyzing supermarket sales data, performing Exploratory Data Analysis (EDA), and training machine learning models to predict sales trends. Built with Python and Tkinter.

## 🚀 Features

-   **User-Friendly Interface**: Modern Dark UI with a dashboard and easy navigation.
-   **Data Management**: Load CSV datasets and view recent files.
-   **Data Preprocessing**:
    -   Handle missing values.
    -   Clean string data.
    -   Encode categorical variables (Label Encoding).
-   **Exploratory Data Analysis (EDA)**:
    -   Generate correlation heatmaps.
    -   Visualise distribution charts.
    -   Analyze categorical vs numerical relationships.
    -   Export charts to PDF.
-   **Machine Learning**:
    -   Train Regression models (Linear Regression, Random Forest, Gradient Boosting) to predict `TotalSale` or `Rating`.
    -   Train Classification models (Random Forest, Gradient Boosting) for categorical targets.
    -   View model performance metrics (R2, MAE, MSE, Accuracy).
-   **Reporting**: Generate PDF reports of your analysis.

## 🛠️ Requirements

Ensure you have Python installed. The application requires the following libraries:

```txt
tkinter
pandas
numpy
matplotlib
seaborn
scikit-learn
Pillow
reportlab
```

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Ahish99/Supermarket-sales-analysis.git
    cd Supermarket-sales-analysis
    ```

2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn Pillow reportlab
    ```

3.  Run the application:
    ```bash
    python sales_app.py
    ```

## 🖥️ Usage

1.  **Login**: Use the default password `12345`.
2.  **Dashboard**: Navigate to different modules using the central hub.
3.  **Load Data**: Upload your supermarket sales CSV file.
4.  **Process**: Clean and encode your data in the Preprocessing section.
5.  **Analyze**: Use EDA tools to visualize trends.
6.  **Model**: Train models to predict outcomes like Sales or Customer Rating.

## 📄 License

This project is open-source and available for educational and analytical purposes.