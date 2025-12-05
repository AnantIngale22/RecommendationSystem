# -*- coding: utf-8 -*-
"""production_data_extractor_fixed.py

Extract data from production PostgreSQL database and dump to recommendation_engine DB
"""

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import warnings
from decimal import Decimal
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pytz
warnings.filterwarnings('ignore')

class KeekDataExtractor:
    def __init__(self, db_config=None):
        # Production database config
        self.production_db_config = db_config or {
            'host': 'localhost',
            'database': 'production_keeks_data',
            'user': 'postgres',
            'password': '1234',
            'port': 5432
        }
        
        # Recommendation engine database config
        self.recommendation_db_config = {
            'host': 'localhost',
            'database': 'recommendation_engine',
            'user': 'postgres',
            'password': '1234',
            'port': 5432
        }
        
        self.connection = None
        self.recommendation_connection = None
        self.lang_encoder = LabelEncoder()
        
    def get_entry_date(self):
        """
        Get entry date in the format: 2023-10-14 14:43:52.294 +0530
        """
        # Create timestamp in IST timezone
        ist = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(ist)
        formatted_date = current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' +0530'
        return formatted_date
        
    def create_recommendation_database(self):
        """
        Create recommendation_engine database if it doesn't exist
        """
        print("🗄️ Creating recommendation_engine database if it doesn't exist...")
        
        try:
            # Connect to default postgres database to create our target database
            temp_config = self.recommendation_db_config.copy()
            temp_config['database'] = 'postgres'  # Connect to default postgres DB
            
            conn = psycopg2.connect(**temp_config)
            conn.autocommit = True  # Must be in autocommit to create database
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'recommendation_engine';")
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute('CREATE DATABASE recommendation_engine;')
                print("✅ Created database 'recommendation_engine'")
            else:
                print("✅ Database 'recommendation_engine' already exists")
                
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to create database: {e}")
            return False
        
    def connect_to_database(self, db_type='production'):
        """Establish connection to PostgreSQL database"""
        try:
            if db_type == 'production':
                self.connection = psycopg2.connect(**self.production_db_config)
                print("✅ Connected to PRODUCTION database successfully!")
            else:
                # Ensure database exists first
                if not self.create_recommendation_database():
                    return False
                    
                self.recommendation_connection = psycopg2.connect(**self.recommendation_db_config)
                print("✅ Connected to RECOMMENDATION_ENGINE database successfully!")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def close_connection(self, db_type='production'):
        """Close database connection"""
        if db_type == 'production' and self.connection:
            self.connection.close()
            print("✅ Production database connection closed")
        elif db_type == 'recommendation' and self.recommendation_connection:
            self.recommendation_connection.close()
            print("✅ Recommendation engine database connection closed")
    
    def execute_query(self, query, params=None, db_type='production'):
        """Execute SQL query and return results as DataFrame"""
        try:
            if db_type == 'production':
                with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params or {})
                    results = cursor.fetchall()
                    df = pd.DataFrame(results)
                    return df
            else:
                with self.recommendation_connection.cursor(cursor_factory=RealDictCursor) as cursor:
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
        
        # Convert decimal columns to float to avoid type issues
        if not interaction_df.empty:
            for col in ['likes', 'views', 'saves']:
                if col in interaction_df.columns:
                    interaction_df[col] = interaction_df[col].astype(float)
            print("   ✅ Converted interaction columns to float")
        
        # Add entry_date to interactions
        entry_date = self.get_entry_date()
        interaction_df['entry_date'] = entry_date
        print(f"   📅 Added entry_date: {entry_date}")
        
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
        
        # Convert age to float and handle decimal types
        if not user_df.empty and 'age' in user_df.columns:
            user_df['age'] = pd.to_numeric(user_df['age'], errors='coerce').astype(float)
        
        # Add entry_date to users
        entry_date = self.get_entry_date()
        user_df['entry_date'] = entry_date
        
        # Show age distribution
        if not user_df.empty:
            valid_ages = user_df['age'].notna().sum()
            total_users = len(user_df)
            print(f"📊 Age extraction stats: {valid_ages}/{total_users} users have valid ages")
            if valid_ages > 0:
                print(f"   Age range: {user_df['age'].min():.0f} - {user_df['age'].max():.0f}")
                print(f"   Average age: {user_df['age'].mean():.1f}")
        
        print(f"✅ Extracted {len(user_df)} users from users_202511031749")
        print(f"   📅 Added entry_date: {entry_date}")
        
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
        
        # Add entry_date to posts
        entry_date = self.get_entry_date()
        post_df['entry_date'] = entry_date
        
        print(f"✅ Extracted {len(post_df)} posts from post_merged_dedup")
        print(f"   📅 Added entry_date: {entry_date}")
        
        return post_df

    def extract_primary_language(self, lang_string):
        """
        Extract primary language from supported_language string
        """
        try:
            if isinstance(lang_string, str) and lang_string.strip():
                clean_string = lang_string.replace('"', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '')
                languages = [lang.strip() for lang in clean_string.split(',') if lang.strip()]
                return languages[0] if languages else 'en'
            return 'en'
        except Exception as e:
            print(f"⚠️ Language parsing error for {lang_string}: {e}")
            return 'en'

    def enhance_user_data(self, user_df):
        """
        Enhance user data with language encoding and other features
        """
        print("🔤 Enhancing user data with language encoding...")
        
        user_enhanced = user_df.copy()
        
        # Extract primary language
        user_enhanced['primary_language'] = user_enhanced['supported_language'].apply(self.extract_primary_language)
        
        # Language encoding
        user_enhanced['lang_encoded'] = self.lang_encoder.fit_transform(
            user_enhanced['primary_language'].fillna('en')
        )
        
        # Show language distribution
        lang_counts = user_enhanced['primary_language'].value_counts()
        print(f"🌍 Language distribution (top 10):")
        for lang, count in lang_counts.head(10).items():
            print(f"   {lang}: {count} users")
        
        print(f"✅ Enhanced user data with {len(lang_counts)} unique languages")
        return user_enhanced

    def enhance_interaction_data(self, interaction_df):
        """
        Enhance interaction data with engagement scoring
        """
        print("🎯 Enhancing interaction data with engagement scores...")
        
        interaction_enhanced = interaction_df.copy()
        
        # Calculate engagement score (weighted combination)
        interaction_enhanced['engagement_score'] = (
            interaction_enhanced['likes'] * 0.5 + 
            interaction_enhanced['views'] * 0.3 + 
            interaction_enhanced['saves'] * 0.2
        )
        
        # Normalize engagement scores
        if interaction_enhanced['engagement_score'].max() > 0:
            interaction_enhanced['engagement_normalized'] = (
                interaction_enhanced['engagement_score'] / interaction_enhanced['engagement_score'].max()
            )
        else:
            interaction_enhanced['engagement_normalized'] = 0
        
        print(f"✅ Enhanced {len(interaction_enhanced)} interactions with engagement scoring")
        return interaction_enhanced

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
        
        if not self.connect_to_database('production'):
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
            self.close_connection('production')

    def save_data_to_csv(self, interaction_df, user_df, post_df, output_dir='data'):
        """
        Save extracted data to CSV files
        """
        print(f"\n💾 Saving data to CSV files in '{output_dir}'...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw data (same as extracted)
        interaction_file = os.path.join(output_dir, 'interaction_table.csv')
        user_file = os.path.join(output_dir, 'user_table.csv')
        post_file = os.path.join(output_dir, 'post_table.csv')
        
        interaction_df.to_csv(interaction_file, index=False)
        user_df.to_csv(user_file, index=False)
        post_df.to_csv(post_file, index=False)
        
        print(f"✅ Raw data saved to CSV files")
        print(f"   📊 Interactions: {interaction_file}")
        print(f"   👤 Users: {user_file}")
        print(f"   🎬 Posts: {post_file}")
        
        return interaction_file, user_file, post_file

    def save_enhanced_data_to_csv(self, enhanced_interaction_df, enhanced_user_df, post_df, output_dir='enhanced_data'):
        """
        Save enhanced data to CSV files
        """
        print(f"\n💾 Saving ENHANCED data to CSV files in '{output_dir}'...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save enhanced data
        enhanced_interaction_file = os.path.join(output_dir, 'enhanced_interaction_table.csv')
        enhanced_user_file = os.path.join(output_dir, 'enhanced_user_table.csv')
        post_file = os.path.join(output_dir, 'post_table.csv')
        
        enhanced_interaction_df.to_csv(enhanced_interaction_file, index=False)
        enhanced_user_df.to_csv(enhanced_user_file, index=False)
        post_df.to_csv(post_file, index=False)
        
        print(f"✅ Enhanced data saved to CSV files")
        print(f"   📊 Enhanced Interactions: {enhanced_interaction_file}")
        print(f"   👤 Enhanced Users: {enhanced_user_file}")
        print(f"   🎬 Posts: {post_file}")
        
        return enhanced_interaction_file, enhanced_user_file, post_file

    def drop_existing_tables(self):
        """
        Drop existing tables to ensure clean schema
        """
        print("🗑️ Dropping existing tables to ensure clean schema...")
        
        try:
            with self.recommendation_connection.cursor() as cursor:
                # Drop tables if they exist (in correct order due to dependencies)
                cursor.execute("DROP TABLE IF EXISTS interactions CASCADE;")
                cursor.execute("DROP TABLE IF EXISTS users CASCADE;")
                cursor.execute("DROP TABLE IF EXISTS posts CASCADE;")
                self.recommendation_connection.commit()
            print("✅ Existing tables dropped successfully")
            return True
        except Exception as e:
            print(f"⚠️ Could not drop tables (might not exist): {e}")
            return True  # Continue anyway

    def create_recommendation_tables(self):
        """
        Create tables in recommendation_engine database if they don't exist
        """
        print("\n🗄️ Creating tables in recommendation_engine database...")
        
        if not self.connect_to_database('recommendation'):
            return False
        
        try:
            # First drop existing tables to ensure clean schema
            self.drop_existing_tables()
            
            # Create users table - ENHANCED with language encoding
            users_table_query = """
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(255) PRIMARY KEY,
                country VARCHAR(100),
                supported_language TEXT,
                age FLOAT,
                primary_language VARCHAR(50),
                lang_encoded INTEGER,
                entry_date VARCHAR(100)
            );
            """
            
            # Create posts table - SAME COLUMNS AS CSV
            posts_table_query = """
            CREATE TABLE IF NOT EXISTS posts (
                post_id VARCHAR(255) PRIMARY KEY,
                post_owner_id VARCHAR(255),
                country VARCHAR(100),
                lang VARCHAR(10),
                entry_date VARCHAR(100)
            );
            """
            
            # Create interactions table - ENHANCED with engagement scoring
            interactions_table_query = """
            CREATE TABLE IF NOT EXISTS interactions (
                user_id VARCHAR(255),
                post_id VARCHAR(255),
                likes FLOAT DEFAULT 0,
                views FLOAT DEFAULT 0,
                saves FLOAT DEFAULT 0,
                engagement_score FLOAT DEFAULT 0,
                engagement_normalized FLOAT DEFAULT 0,
                entry_date VARCHAR(100),
                UNIQUE(user_id, post_id)
            );
            """
            
            with self.recommendation_connection.cursor() as cursor:
                cursor.execute(users_table_query)
                cursor.execute(posts_table_query)
                cursor.execute(interactions_table_query)
                self.recommendation_connection.commit()
            
            print("✅ Tables created successfully in recommendation_engine database!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create tables: {e}")
            self.recommendation_connection.rollback()
            return False
        finally:
            self.close_connection('recommendation')

    def _safe_float_conversion(self, value):
        """Safely convert any value to float, handling Decimal and other types"""
        try:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, Decimal):
                return float(value)
            return float(str(value))
        except (ValueError, TypeError):
            return 0.0

    def dump_data_to_recommendation_db(self, interaction_df, user_df, post_df):
        """
        Dump ENHANCED data to recommendation_engine database
        """
        print("\n🚀 Dumping ENHANCED data to recommendation_engine database...")
        
        if not self.connect_to_database('recommendation'):
            print("❌ Cannot connect to recommendation_engine database")
            return None, None
        
        try:
            # First create tables if they don't exist (this now includes dropping old ones)
            if not self.create_recommendation_tables():
                return None, None
            
            # Enhance the data
            enhanced_user_df = self.enhance_user_data(user_df)
            enhanced_interaction_df = self.enhance_interaction_data(interaction_df)
            
            # Reconnect for data insertion
            self.connect_to_database('recommendation')
            
            with self.recommendation_connection.cursor() as cursor:
                # Insert ENHANCED user data
                print("👤 Inserting ENHANCED user data...")
                user_count = 0
                for _, row in enhanced_user_df.iterrows():
                    cursor.execute(
                        """INSERT INTO users (user_id, country, supported_language, age, primary_language, lang_encoded, entry_date) 
                         VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        (
                            str(row['user_id']), 
                            str(row['country']), 
                            str(row['supported_language']), 
                            self._safe_float_conversion(row['age']),
                            str(row['primary_language']),
                            int(row['lang_encoded']),
                            str(row['entry_date'])
                        )
                    )
                    user_count += 1
                print(f"   ✅ Inserted {user_count} enhanced users")
                
                # Insert post data
                print("🎬 Inserting post data...")
                post_count = 0
                for _, row in post_df.iterrows():
                    cursor.execute(
                        "INSERT INTO posts (post_id, post_owner_id, country, lang, entry_date) VALUES (%s, %s, %s, %s, %s)",
                        (
                            str(row['post_id']), 
                            str(row['post_owner_id']), 
                            str(row['country']), 
                            str(row['lang']),
                            str(row['entry_date'])
                        )
                    )
                    post_count += 1
                print(f"   ✅ Inserted {post_count} posts")
                
                # Insert ENHANCED interaction data
                print("📊 Inserting ENHANCED interaction data...")
                interaction_count = 0
                for _, row in enhanced_interaction_df.iterrows():
                    cursor.execute(
                        """INSERT INTO interactions (user_id, post_id, likes, views, saves, engagement_score, engagement_normalized, entry_date) 
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            str(row['user_id']), 
                            str(row['post_id']), 
                            self._safe_float_conversion(row['likes']),
                            self._safe_float_conversion(row['views']),
                            self._safe_float_conversion(row['saves']),
                            self._safe_float_conversion(row['engagement_score']),
                            self._safe_float_conversion(row['engagement_normalized']),
                            str(row['entry_date'])
                        )
                    )
                    interaction_count += 1
                    
                    # Show progress for large datasets
                    if interaction_count % 100000 == 0:
                        print(f"   📦 Processed {interaction_count:,} interactions...")
                
                print(f"   ✅ Inserted {interaction_count:,} enhanced interactions")
                
                self.recommendation_connection.commit()
            
            # Verify data insertion
            with self.recommendation_connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM posts")
                post_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM interactions")
                interaction_count = cursor.fetchone()[0]
            
            print(f"✅ ENHANCED data dumped successfully to recommendation_engine database!")
            print(f"📊 Verification counts:")
            print(f"   👤 Enhanced Users: {user_count:,}")
            print(f"   🎬 Posts: {post_count:,}")
            print(f"   📊 Enhanced Interactions: {interaction_count:,}")
            
            return enhanced_interaction_df, enhanced_user_df
            
        except Exception as e:
            print(f"❌ Failed to dump data: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            self.recommendation_connection.rollback()
            return None, None
        finally:
            self.close_connection('recommendation')

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
        # Save RAW data to CSV
        extractor.save_data_to_csv(interaction_df, user_df, post_df)
        
        # Dump ENHANCED data to recommendation database
        enhanced_interaction_df, enhanced_user_df = extractor.dump_data_to_recommendation_db(interaction_df, user_df, post_df)
        
        # Save ENHANCED data to CSV
        if enhanced_interaction_df is not None:
            extractor.save_enhanced_data_to_csv(enhanced_interaction_df, enhanced_user_df, post_df)
        
        print(f"\n🎉 Complete data extraction and dumping completed successfully!")
        print(f"📊 Total records processed:")
        print(f"   - Interactions: {len(interaction_df):,}")
        print(f"   - Users: {len(user_df):,}")
        print(f"   - Posts: {len(post_df):,}")
    else:
        print("❌ Data extraction failed")

if __name__ == "__main__":
    main()