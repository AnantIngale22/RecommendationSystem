import pandas as pd
import os

def analyze_log():
    log_file = "data/interaction_log.csv"
    
    if not os.path.exists(log_file):
        print("❌ No interaction log found.")
        return

    print("📊 Loading Interaction Log...")
    df = pd.read_csv(log_file)
    
    # Count actions per user
    user_stats = df.groupby(['user_id', 'action']).size().unstack(fill_value=0)
    
    # Add Total column
    user_stats['Total'] = user_stats.sum(axis=1)
    
    # Sort by Total Activity
    user_stats = user_stats.sort_values('Total', ascending=False)
    
    print("\n👤 USER ACTIVITY REPORT")
    print("=" * 60)
    print(f"{'User ID':<15} | {'Like':<5} | {'View':<5} | {'Save':<5} | {'Total':<5}")
    print("-" * 60)
    
    for user_id, row in user_stats.iterrows():
        # Handle missing columns if they don't exist in the data
        like = row.get('like', 0)
        view = row.get('view', 0)
        save = row.get('save', 0)
        total = row['Total']
        
        print(f"{user_id:<15} | {like:<5} | {view:<5} | {save:<5} | {total:<5}")
        
    print("=" * 60)
    print(f"✅ Total Interactions: {len(df)}")

if __name__ == "__main__":
    analyze_log()
