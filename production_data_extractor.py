# -*- coding: utf-8 -*-
"""production_data_extractor_fixed.py

Extract data from production PostgreSQL database for recommendation system
USING UPDATED TABLE NAMES
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import warnings
warnings.filterwarnings('ignore')

class KeekDataExtractor:
    def __init__(self, db_config=None):
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'production_keeks_data',
            'user': 'postgres',
            'password': '1234',
            'port': 5432
        }
        self.connection = None
        
    def connect_to_database(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            print("✅ Connected to PostgreSQL database successfully!")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✅ Database connection closed")
    
    def execute_query(self, query, params=None):
        """Execute SQL query and return results as DataFrame"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params or {})
                results = cursor.fetchall()
                df = pd.DataFrame(results)
                return df
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return pd.DataFrame()

    def extract_interaction_data(self):
        """
        Extract ALL user interaction data from stream_viewer_merged table
        """
        print("📊 Extracting ALL interaction data...")
        
        query = """
        SELECT 
            user_id,
            post_id,
            SUM(COALESCE(likes, 0)) AS likes,
            SUM(COALESCE(view_count, 0)) AS views,
            SUM(CASE WHEN saved = 'Y' THEN 1 ELSE 0 END) AS saves
        FROM 
            public.stream_viewer_merged
        WHERE 
            post_id IS NOT NULL
            AND user_id IS NOT NULL
        GROUP BY 
            user_id, post_id
        HAVING 
            SUM(COALESCE(likes, 0)) > 0 
            OR SUM(COALESCE(view_count, 0)) > 0
        ORDER BY 
            user_id, post_id;
        """
        
        interaction_df = self.execute_query(query)
        print(f"✅ Extracted {len(interaction_df)} interactions from stream_viewer_merged")
        return interaction_df
    
    def extract_user_data(self, user_ids=None):
        """
        Extract user demographic data from users_202511031749 table
        """
        print("👤 Extracting user data from users_202511031749 table...")
        
        if user_ids:
            user_filter = "AND u.id IN %s"
            params = (tuple(user_ids),)
        else:
            user_filter = ""
            params = ()
        
        query = f"""
        SELECT DISTINCT
            u.id AS user_id,
            u.country,
            cm.supported_languages::text AS supported_language,
            CASE 
                WHEN u.birthday IS NOT NULL 
                THEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, u.birthday::timestamp))
                ELSE NULL 
            END AS age
        FROM 
            public.users_202511031749 u
        INNER JOIN (
            SELECT DISTINCT user_id
            FROM public.stream_viewer_merged 
            WHERE post_id IS NOT NULL
            AND (likes > 0 OR view_count > 0)
        ) sv ON u.id = sv.user_id
        LEFT JOIN 
            public.country_matrix_202511031750 cm ON cm.country = u.country
        WHERE 
            u.country IS NOT NULL 
            AND u.country != ''
            {user_filter}
        ORDER BY 
            u.id;
        """
        
        user_df = self.execute_query(query, params)
        
        # Show age distribution
        if not user_df.empty:
            valid_ages = user_df['age'].notna().sum()
            total_users = len(user_df)
            print(f"📊 Age extraction stats: {valid_ages}/{total_users} users have valid ages")
            if valid_ages > 0:
                print(f"   Age range: {user_df['age'].min():.0f} - {user_df['age'].max():.0f}")
                print(f"   Average age: {user_df['age'].mean():.1f}")
        
        print(f"✅ Extracted {len(user_df)} users from users_202511031749")
        return user_df
    
    def extract_post_data(self, post_ids=None):
        """
        Extract post/video metadata from post_merged_dedup table
        """
        print("🎬 Extracting post data from post_merged_dedup...")
        
        if post_ids:
            post_filter = "AND p.post_id IN %s"
            params = (tuple(post_ids),)
        else:
            post_filter = ""
            params = ()
        
        query = f"""
        SELECT DISTINCT
            p.post_id,
            p.user_id AS post_owner_id,
            COALESCE(NULLIF(p.country, ''), 'US') AS country,
            COALESCE(NULLIF(p.lang, ''), 'en') AS lang
        FROM
            public.post_merged_dedup p
        WHERE
            p.post_id IN (
                SELECT DISTINCT post_id
                FROM public.stream_viewer_merged
                WHERE post_id IS NOT NULL
            )
            {post_filter}
        ORDER BY
            p.post_id;
        """
        
        post_df = self.execute_query(query, params)
        print(f"✅ Extracted {len(post_df)} posts from post_merged_dedup")
        return post_df

    def extract_complete_dataset(self, sample_size=None):
        """
        Extract COMPLETE dataset for recommendation system
        """
        print("🚀 Starting COMPLETE data extraction...")
        print("="*50)
        print("📊 USING UPDATED TABLE NAMES:")
        print("   - stream_viewer_merged")
        print("   - users_202511031749") 
        print("   - post_merged_dedup")
        print("   - country_matrix_202511031750")
        print("="*50)
        
        if not self.connect_to_database():
            return None, None, None
        
        try:
            # Step 1: Extract ALL interactions
            interaction_df = self.extract_interaction_data()
            
            if interaction_df.empty:
                print("❌ No interaction data found")
                return None, None, None
            
            # Apply sampling if requested
            if sample_size and len(interaction_df) > sample_size:
                interaction_df = interaction_df.sample(sample_size, random_state=42)
                print(f"📝 Sampled {sample_size} interactions")
            
            # Step 2: Get unique users and posts from interactions
            unique_users = interaction_df['user_id'].unique().tolist()
            unique_posts = interaction_df['post_id'].unique().tolist()
            
            print(f"🔍 Found {len(unique_users)} unique users and {len(unique_posts)} unique posts")
            
            # Step 3: Extract user and post data
            user_df = self.extract_user_data(unique_users)
            post_df = self.extract_post_data(unique_posts)
            
            print("\n🎉 COMPLETE DATA EXTRACTION COMPLETED!")
            print("="*50)
            print(f"📊 Final Dataset Sizes:")
            print(f"   Interactions: {len(interaction_df)} (from stream_viewer_merged)")
            print(f"   Users: {len(user_df)} (from users_202511031749)")
            print(f"   Posts: {len(post_df)} (from post_merged_dedup)")
            
            return interaction_df, user_df, post_df
            
        finally:
            self.close_connection()

    def save_data_to_csv(self, interaction_df, user_df, post_df, output_dir='data'):
        """
        Save extracted data to CSV files
        """
        print(f"\n💾 Saving data to CSV files in '{output_dir}'...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        interaction_file = os.path.join(output_dir, 'interaction_table.csv')
        user_file = os.path.join(output_dir, 'user_table.csv')
        post_file = os.path.join(output_dir, 'post_table.csv')
        
        interaction_df.to_csv(interaction_file, index=False)
        user_df.to_csv(user_file, index=False)
        post_df.to_csv(post_file, index=False)
        
        print(f"✅ Interactions saved to: {interaction_file}")
        print(f"✅ Users saved to: {user_file}")
        print(f"✅ Posts saved to: {post_file}")
        
        return interaction_file, user_file, post_file

def main():
    """Main function to run data extraction"""
    print("🎯 KEek COMPLETE DATA EXTRACTOR (UPDATED TABLE NAMES)")
    print("="*50)
    
    # Initialize extractor with your actual credentials
    extractor = KeekDataExtractor()
    
    # Extract ALL data (remove sample_size for complete dataset)
    interaction_df, user_df, post_df = extractor.extract_complete_dataset(
        sample_size=None  # Remove this parameter for ALL data
    )
    
    if interaction_df is not None:
        # Save to CSV
        extractor.save_data_to_csv(interaction_df, user_df, post_df)
        
        # Show sample data
        print("\n📋 SAMPLE DATA:")
        print("Interactions:")
        print(interaction_df[['user_id', 'post_id', 'likes', 'views']].head(3))
        print("\nUsers (with ages):")
        print(user_df[['user_id', 'country', 'supported_language', 'age']].head(10))
        print("\nPosts:")
        print(post_df.head(3))
        
        print(f"\n🎉 Complete data extraction completed successfully!")
        print(f"📊 Total records extracted:")
        print(f"   - Interactions: {len(interaction_df)}")
        print(f"   - Users: {len(user_df)}")
        print(f"   - Posts: {len(post_df)}")
    else:
        print("❌ Data extraction failed")

if __name__ == "__main__":
    main()