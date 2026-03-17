import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of records
n_records = 50000

# Generate data
order_ids = range(1, n_records + 1)
order_dates = [datetime(2022, 1, 1) + timedelta(days=random.randint(0, 730)) for _ in range(n_records)]
product_ids = np.random.randint(1000, 10000, n_records)
categories = ['Books', 'Fashion', 'Sports', 'Beauty', 'Electronics', 'Home & Garden']
product_categories = np.random.choice(categories, n_records)
prices = np.random.uniform(10, 1000, n_records).round(2)
discount_percents = np.random.choice([0, 5, 10, 15, 20, 25, 30], n_records)
quantities = np.random.randint(1, 10, n_records)
regions = ['North America', 'Asia', 'Europe', 'Middle East']
customer_regions = np.random.choice(regions, n_records)
payment_methods = ['UPI', 'Credit Card', 'Debit Card', 'PayPal', 'Wallet', 'Cash']
payment_methods_choice = np.random.choice(payment_methods, n_records)
ratings = np.random.uniform(1, 5, n_records).round(1)
review_counts = np.random.randint(100, 2000, n_records)

# Calculate derived columns
discounted_prices = prices * (1 - discount_percents / 100)
total_revenues = discounted_prices * quantities

# Create DataFrame
df = pd.DataFrame({
    'order_id': order_ids,
    'order_date': [d.strftime('%Y-%m-%d') for d in order_dates],
    'product_id': product_ids,
    'product_category': product_categories,
    'price': prices,
    'discount_percent': discount_percents,
    'quantity_sold': quantities,
    'customer_region': customer_regions,
    'payment_method': payment_methods_choice,
    'rating': ratings,
    'review_count': review_counts,
    'discounted_price': discounted_prices.round(2),
    'total_revenue': total_revenues.round(2)
})

# Save to CSV
df.to_csv('sales_data.csv', index=False)
print(f"Generated {len(df)} records and saved to sales_data.csv")