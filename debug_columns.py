import pandas as pd

print("🔍 DEBUGGING YOUR DATA COLUMNS...")

# Load all your data files
print("\n📋 USER TABLE:")
user_df = pd.read_csv('data/user_table_poc_output.csv')
print(f"Columns: {user_df.columns.tolist()}")
print(f"Shape: {user_df.shape}")
print("First 2 rows:")
print(user_df.head(2))

print("\n📋 INTERACTION TABLE:")
interaction_df = pd.read_csv('data/interaction_table_poc_output.csv')
print(f"Columns: {interaction_df.columns.tolist()}")
print(f"Shape: {interaction_df.shape}")
print("First 2 rows:")
print(interaction_df.head(2))

print("\n📋 POST TABLE:")
post_df = pd.read_csv('data/post_table_poc_output.csv')
print(f"Columns: {post_df.columns.tolist()}")
print(f"Shape: {post_df.shape}")
print("First 2 rows:")
print(post_df.head(2))