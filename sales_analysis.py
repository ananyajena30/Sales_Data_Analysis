import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 5)
sns.set_style("whitegrid")

df = pd.read_csv("sales_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

df['Order_Date'] = pd.to_datetime(df['Order_Date'])

df['Revenue'] = df['Quantity'] * df['Price']

df['Month'] = df['Order_Date'].dt.month
df['Year'] = df['Order_Date'].dt.year

total_revenue = df['Revenue'].sum()
print("\nTotal Revenue:", total_revenue)

top_products = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
print("\nTop Products:")
print(top_products.head(10))
monthly_sales = df.groupby('Month')['Revenue'].sum()
print("\nMonthly Sales:")
print(monthly_sales)

city_sales = df.groupby('City')['Revenue'].sum().sort_values(ascending=False)
print("\nCity Sales:")
print(city_sales)

category_sales = df.groupby('Category')['Revenue'].sum()
print("\nCategory Sales:")
print(category_sales)
plt.figure()
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.show()

plt.figure()
top_products.head(5).plot(kind='bar')
plt.title("Top 5 Selling Products")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45)
plt.show()

plt.figure()
sns.barplot(x=city_sales.index, y=city_sales.values)
plt.title("Sales by City")
plt.xlabel("City")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.show()

plt.figure()
category_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title("Sales by Category")
plt.ylabel("")
plt.show()

df['Profit'] = df['Revenue'] * 0.2

monthly_profit = df.groupby('Month')['Profit'].sum()

plt.figure()
monthly_profit.plot(marker='o', linestyle='--')
plt.title("Monthly Profit Trend")
plt.xlabel("Month")
plt.ylabel("Profit")
plt.show()

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

df.to_csv("cleaned_sales_data.csv", index=False)

print("\n✅ Analysis Complete! Cleaned data saved.")
