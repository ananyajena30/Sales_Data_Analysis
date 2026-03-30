import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 5)
sns.set_style("whitegrid")


# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("sales_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# ==============================
# 3. Data Cleaning
# ==============================
# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Drop missing values
df = df.dropna()

# Convert Order_Date to datetime
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# ==============================
# 4. Feature Engineering
# ==============================
# Create Revenue column
df['Revenue'] = df['Quantity'] * df['Price']

# Extract Month & Year
df['Month'] = df['Order_Date'].dt.month
df['Year'] = df['Order_Date'].dt.year

# ==============================
# 5. Data Analysis
# ==============================

# Total Revenue
total_revenue = df['Revenue'].sum()
print("\nTotal Revenue:", total_revenue)

# Top Selling Products
top_products = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
print("\nTop Products:")
print(top_products.head(10))


# Monthly Sales
monthly_sales = df.groupby('Month')['Revenue'].sum()
print("\nMonthly Sales:")
print(monthly_sales)

# Sales by City
city_sales = df.groupby('City')['Revenue'].sum().sort_values(ascending=False)
print("\nCity Sales:")
print(city_sales)

# Category-wise Sales
category_sales = df.groupby('Category')['Revenue'].sum()
print("\nCategory Sales:")
print(category_sales)

# ==============================
# 6. Data Visualization
# ==============================

# 1. Monthly Sales Trend
plt.figure()
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.show()

# 2. Top 5 Products
plt.figure()
top_products.head(5).plot(kind='bar')
plt.title("Top 5 Selling Products")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45)
plt.show()

# 3. Sales by City
plt.figure()
sns.barplot(x=city_sales.index, y=city_sales.values)
plt.title("Sales by City")
plt.xlabel("City")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()

# 4. Category-wise Sales Pie Chart
plt.figure()
category_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title("Sales by Category")
plt.ylabel("")
plt.show()

# ==============================
# 7. Advanced Analysis
# ==============================

# Profit (Assume 20%)
df['Profit'] = df['Revenue'] * 0.2

# Monthly Profit
monthly_profit = df.groupby('Month')['Profit'].sum()

plt.figure()
monthly_profit.plot(marker='o', linestyle='--')
plt.title("Monthly Profit Trend")
plt.xlabel("Month")
plt.ylabel("Profit")
plt.show()

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# 8. Save Clean Data
# ==============================
df.to_csv("cleaned_sales_data.csv", index=False)

print("\n✅ Analysis Complete! Cleaned data saved.")
