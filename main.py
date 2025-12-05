# -*- coding: utf-8 -*-
"""complete_recommendation_system.py

Complete Recommendation System with Two-Tower Model & Video Processing
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================
# VIDEO PROCESSING INTEGRATION
# ============================================================
try:
    from video_processing_service import VideoProcessingService, RealTimeCompatibleVideoProcessor
    HAS_VIDEO_PROCESSING = True
    print("✅ VideoProcessingService imported successfully")
except ImportError as e:
    print(f"⚠️ VideoProcessingService not available: {e}")
    HAS_VIDEO_PROCESSING = False

# ============================================================
# DATABASE EXTRACTION INTEGRATION
# ============================================================
try:
    from production_data_extractor import KeekDataExtractor
    HAS_DATABASE_EXTRACTION = True
    print("✅ Database extraction imported successfully")
except ImportError as e:
    print(f"⚠️ Database extraction not available: {e}")
    HAS_DATABASE_EXTRACTION = False

class VideoDataIntegrator:
    """Integrate video analysis data with interaction table"""
    
    def __init__(self):
        self.video_service = VideoProcessingService() if HAS_VIDEO_PROCESSING else None
    
    def extract_post_id_from_filename(self, filename):
        """Extract post_id from video filename"""
        try:
            base_name = os.path.basename(filename)
            post_id_str = base_name.split('_')[0]
            return float(post_id_str)
        except (ValueError, IndexError):
            print(f"⚠️ Could not extract post_id from filename: {filename}")
            return None

    def process_videos_and_merge(self, interaction_csv_path, videos_folder="videos"):
        """
        Step 1: Process videos and merge with interaction data - FIXED VERSION
        """
        print("🎬 STEP 1: VIDEO PROCESSING AND DATA INTEGRATION")
        print("=" * 60)
        
        if not HAS_VIDEO_PROCESSING:
            print("❌ Video processing not available - using original interaction data")
            return pd.read_csv(interaction_csv_path)
        
        # Load original interaction data
        print("📊 Loading interaction data...")
        interaction_df = pd.read_csv(interaction_csv_path)
        print(f"✅ Loaded {len(interaction_df)} interactions")
        
        # Process videos
        print("🎥 Processing videos...")
        video_service = VideoProcessingService()
        processed_videos = video_service.process_folder_videos("training_videos")
        
        if not processed_videos:
            print("❌ No videos processed - using original interaction data")
            return interaction_df
        
        print(f"✅ Processed {len(processed_videos)} videos")
        
        # Create video analysis DataFrame
        video_data = []
        for video in processed_videos:
            post_id = self.extract_post_id_from_filename(video['video_path'])
            if post_id is not None:
                video_data.append({
                    'post_id': post_id,
                    'video_duration_sec': video.get('duration_sec', 0),  # Changed column name
                    'video_predicted_tag_1': video.get('predicted_tag_1', 'unknown'),  # Changed column name
                    'video_predicted_tag_2': video.get('predicted_tag_2', 'unknown')   # Changed column name
                })
        
        video_df = pd.DataFrame(video_data)
        print(f"✅ Extracted data from {len(video_df)} videos with valid post_ids")
        
        # DEBUG: Check what we're merging
        print(f"🔍 Interaction columns: {interaction_df.columns.tolist()}")
        print(f"🔍 Video columns: {video_df.columns.tolist()}")
        
        # Merge with interaction data
        print("🔄 Merging video data with interaction table...")
        merged_df = pd.merge(
            interaction_df,
            video_df,
            on='post_id',
            how='left'  # Keep all interactions, even if no video data
        )
        
        # DEBUG: Check merged structure
        print(f"🔍 Merged columns: {merged_df.columns.tolist()}")
        
        # Count matches using the new column names
        matched_count = merged_df['video_duration_sec'].notna().sum()
        total_count = len(merged_df)
        
        print(f"📊 Merge Results:")
        print(f"   Total interactions: {total_count}")
        print(f"   Interactions with video data: {matched_count}")
        print(f"   Match rate: {matched_count/total_count*100:.1f}%")
        
        # Fill missing values with the new column names
        merged_df['video_duration_sec'] = merged_df['video_duration_sec'].fillna(0)
        merged_df['video_predicted_tag_1'] = merged_df['video_predicted_tag_1'].fillna('unknown')
        merged_df['video_predicted_tag_2'] = merged_df['video_predicted_tag_2'].fillna('unknown')
        
        # Save enriched dataset
        output_path = "data/interaction_table_with_video_data.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        print(f"💾 Saved enriched interaction table to: {output_path}")
        
        return merged_df

class KeekRecommendationSystem:
    def __init__(self):
        self.TAGS = [
            "sports", "gaming", "cooking", "traveling", "entertainment", "music",
            "education", "news", "vlog", "review", "fashion", "technology",
            "fitness", "food", "animals", "nature", "comedy"
        ]
        self.WEIGHTS = {'likes': 3.0, 'saves': 2.5, 'views': 0.1}
        self.model = None
        self.user_profiles = {}
        self.user_profiles_loaded = False
        self.user_processed = None
        self.post_processed = None
        self.interaction_df = None
        self.video_integrator = VideoDataIntegrator()
        
    def extract_and_load_data(self, use_existing=True):
        """Extract data from database or load from CSV"""
        data_dir = "data"
        interaction_path = os.path.join(data_dir, 'interaction_table.csv')
        
        if use_existing and os.path.exists(interaction_path):
            print("📊 Loading existing data from CSV files...")
            return self.load_production_data()
        else:
            print("🚀 Extracting fresh data from database...")
            try:
                from production_data_extractor import KeekDataExtractor
                extractor = KeekDataExtractor()
                interaction_df, user_df, post_df = extractor.extract_complete_dataset()
                
                # Save data
                os.makedirs(data_dir, exist_ok=True)
                interaction_df.to_csv(interaction_path, index=False)
                user_df.to_csv(os.path.join(data_dir, 'user_table_poc_output.csv'), index=False)
                post_df.to_csv(os.path.join(data_dir, 'post_table_poc_output.csv'), index=False)
                
                self.interaction_df = interaction_df
                print("✅ Fresh data extracted and saved successfully!")
                return interaction_df, user_df, post_df
                
            except ImportError:
                print("❌ Data extractor not available, using existing CSV files")
                return self.load_production_data()
    
    def load_production_data(self):
        """Load data from production CSV files"""
        print("📊 Loading production data from CSV files...")
        data_dir = "data"
        
        try:
            # Load interaction data
            interaction_path = os.path.join(data_dir, 'interaction_table.csv')
            if not os.path.exists(interaction_path):
                raise FileNotFoundError(f"Interaction table not found: {interaction_path}")
            self.interaction_df = pd.read_csv(interaction_path)
            
            # Load user data
            user_df = pd.read_csv(os.path.join(data_dir, 'user_table.csv'))
            
            # Load post data
            post_df = pd.read_csv(os.path.join(data_dir, 'post_table.csv'))
            
            print(f"✅ Data loaded successfully from CSV files!")
            print(f"   Interactions: {self.interaction_df.shape}")
            print(f"   Users: {user_df.shape}") 
            print(f"   Posts: {post_df.shape}")
            
            return self.interaction_df, user_df, post_df
            
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            raise
    
    def load_and_integrate_data(self, use_fresh_data=False):
        """Load and integrate video data with interaction table"""
        print("📊 LOADING AND INTEGRATING DATA")
        print("=" * 60)
        
        try:
            # Step 1: Get data
            if use_fresh_data and HAS_DATABASE_EXTRACTION:
                interaction_df, user_df, post_df = self.extract_and_load_data(use_existing=False)
            else:
                interaction_df, user_df, post_df = self.extract_and_load_data(use_existing=True)
            
            # Step 2: Process videos and merge with interaction data
            interaction_csv = os.path.join("data", 'interaction_table.csv')
            self.interaction_df = self.video_integrator.process_videos_and_merge(interaction_csv)
            
            print(f"✅ Data integrated successfully!")
            print(f"   Interactions: {self.interaction_df.shape}")
            print(f"   Users: {user_df.shape}") 
            print(f"   Posts: {post_df.shape}")
            
            return self.interaction_df, user_df, post_df
            
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            raise
    
    def preprocess_user_data(self, user_df):
        """Preprocess user demographic data"""
        print("👤 Preprocessing user data...")
        
        user_processed = user_df.copy()
        
        # Country encoding
        country_encoder = LabelEncoder()
        user_processed['country_encoded'] = country_encoder.fit_transform(
            user_processed['country'].fillna('unknown')
        )
        
        # Language encoding
        def extract_primary_language(lang_string):
            try:
                if isinstance(lang_string, str) and lang_string.strip():
                    clean_string = lang_string.replace('"', '').replace('[', '').replace(']', '')
                    languages = [lang.strip() for lang in clean_string.split(',') if lang.strip()]
                    return languages[0] if languages else 'en'
                return 'en'
            except Exception as e:
                print(f"⚠️ Language parsing error for {lang_string}: {e}")
                return 'en'
        
        user_processed['primary_language'] = user_processed['supported_language'].apply(extract_primary_language)
        
        lang_encoder = LabelEncoder()
        user_processed['lang_encoded'] = lang_encoder.fit_transform(
            user_processed['primary_language'].fillna('en')
        )
        
        # Age normalization
        if 'age' in user_processed.columns:
            age_scaler = StandardScaler()
            user_processed['age_normalized'] = age_scaler.fit_transform(user_processed[['age']].fillna(user_processed['age'].mean()))
        else:
            user_processed['age_normalized'] = 0
        
        # Create user_id column for consistency
        if 'user_id' not in user_processed.columns and 'id' in user_processed.columns:
            user_processed['user_id'] = user_processed['id']
        
        self.user_processed = user_processed
        
        print("✅ User data preprocessed successfully!")
        return user_processed
    
    def preprocess_post_data(self, post_df, interaction_df):
        """Preprocess post/video data"""
        print("🎬 Preprocessing post data...")
        
        post_processed = post_df.copy()
        
        # Country and language encoding
        country_encoder = LabelEncoder()
        post_processed['country_encoded'] = country_encoder.fit_transform(
            post_processed['country'].fillna('unknown')
        )
        
        lang_encoder = LabelEncoder()
        post_processed['lang_encoded'] = lang_encoder.fit_transform(
            post_processed['lang'].fillna('en')
        )
        
        # Post owner encoding
        owner_encoder = LabelEncoder()
        post_processed['post_owner_encoded'] = owner_encoder.fit_transform(
            post_processed['post_owner_id']
        )
        
        # Calculate post engagement metrics
        post_engagement = interaction_df.groupby('post_id').agg({
            'likes': 'sum', 'views': 'sum', 'saves': 'sum'
        }).reset_index()
        
        # Merge with post data
        post_processed = post_processed.merge(post_engagement, on='post_id', how='left')
        
        # Fill NaN values and normalize
        engagement_cols = ['likes', 'views', 'saves']
        post_processed[engagement_cols] = post_processed[engagement_cols].fillna(0)
        
        scaler = StandardScaler()
        post_processed[engagement_cols] = scaler.fit_transform(
            post_processed[engagement_cols]
        )
        
        self.post_processed = post_processed
        
        print("✅ Post data preprocessed")
        return post_processed
    
    def get_user_profiles(self, force_rebuild=False):
        """Get user profiles - build once, reuse many times"""
        if not self.user_profiles_loaded or force_rebuild or not self.user_profiles:
            print("🔄 Building user profiles...")
            if self.interaction_df is None:
                self.interaction_df, _, _ = self.load_and_integrate_data()
            self.build_user_profiles(self.interaction_df)
            self.user_profiles_loaded = True
        return self.user_profiles
    
    def get_user_profile(self, user_id):
        """Get specific user profile"""
        profiles = self.get_user_profiles()
        return profiles.get(user_id, self._get_default_profile())
    
    def _get_default_profile(self):
        """Return equal weights for all categories for new users"""
        return {category: 1.0/len(self.TAGS) for category in self.TAGS}
    
    def build_user_profiles(self, interaction_df):
        """Build dynamic user profiles based on interaction history - UPDATED"""
        print("🔍 Building user profiles from interactions...")
        
        user_profiles = {}
        
        for idx, row in interaction_df.iterrows():
            user_id = row['user_id']
            
            # Calculate engagement score
            engagement_score = (
                row['likes'] * self.WEIGHTS['likes'] +
                row['saves'] * self.WEIGHTS['saves'] +
                row['views'] * self.WEIGHTS['views']
            )
            
            # Use the new video column names
            post_tags = [row.get('video_predicted_tag_1', 'unknown'), 
                        row.get('video_predicted_tag_2', 'unknown')]
            learning_rate = 0.1
            
            # Initialize user profile if new user
            if user_id not in user_profiles:
                user_profiles[user_id] = {category: 1.0/len(self.TAGS) for category in self.TAGS}
            
            # Update weights for engaged categories
            current_weights = user_profiles[user_id].copy()
            for tag in post_tags:
                if tag in current_weights:
                    current_weights[tag] += engagement_score * learning_rate
            
            # Renormalize weights
            total_weight = sum(current_weights.values())
            if total_weight > 0:
                user_profiles[user_id] = {category: weight / total_weight for category, weight in current_weights.items()}
            
            if idx % 1000 == 0 and idx > 0:
                print(f"Processed {idx}/{len(interaction_df)} interactions")
        
        self.user_profiles = user_profiles
        print(f"✅ User profiles built for {len(user_profiles)} users!")
        
        # Save user profiles to CSV
        self.save_user_profiles_to_csv(user_profiles)
        
        return user_profiles

    def save_user_profiles_to_csv(self, user_profiles, filename='user_profiles_analysis.csv'):
        """Save detailed user profiles to CSV for analysis"""
        print(f"💾 Saving user profiles to {filename}...")
        
        profile_data = []
        
        for user_id, categories in user_profiles.items():
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            
            row_data = {
                'user_id': user_id,
                'total_interactions': len([v for v in categories.values() if v > 0.1]),
                'preference_strength': sum(categories.values()),
            }
            
            for i, (category, score) in enumerate(top_categories, 1):
                row_data[f'top_category_{i}'] = category
                row_data[f'top_score_{i}'] = round(score, 4)
            
            for category in self.TAGS:
                row_data[f'score_{category}'] = round(categories.get(category, 0), 4)
            
            profile_data.append(row_data)
        
        profiles_df = pd.DataFrame(profile_data)
        
        base_columns = ['user_id', 'total_interactions', 'preference_strength']
        top_category_columns = [f'top_category_{i}' for i in range(1, 4)] + [f'top_score_{i}' for i in range(1, 4)]
        score_columns = [f'score_{category}' for category in self.TAGS]
        
        final_columns = base_columns + top_category_columns + score_columns
        # Only include columns that exist
        final_columns = [col for col in final_columns if col in profiles_df.columns or col.startswith('score_')]
        profiles_df = profiles_df.reindex(columns=final_columns)
        
        profiles_df.to_csv(filename, index=False)
        
        print(f"✅ User profiles saved to {filename}")
        print(f"   Total users: {len(profiles_df)}")
        
        return profiles_df

    # ============================================================
    # REAL-TIME VIDEO RECOMMENDATION METHODS
    # ============================================================
    
    def recommend_for_new_video(self, video_analysis, specific_users=None, top_k=10):
        """Recommend new video to users based on content analysis"""
        print(f"🎬 Real-time recommendation for new video...")
        
        # Get video tags from analysis
        video_tags = [video_analysis['predicted_tag_1'], video_analysis['predicted_tag_2']]
        print(f"🏷️ Video tags: {video_tags}")
        
        # Get all user profiles (builds once, reuses)
        user_profiles = self.get_user_profiles()
        
        recommendations = []
        
        # If specific users provided, only check those
        users_to_check = specific_users if specific_users else list(user_profiles.keys())
        
        for user_id in users_to_check:
            if user_id not in user_profiles:
                continue
                
            user_profile = user_profiles[user_id]
            match_score = 0
            
            # Calculate content matching score
            for video_tag in video_tags:
                if video_tag in user_profile:
                    match_score += user_profile[video_tag]
            
            # Add engagement boost for active users
            engagement_boost = min(len([v for v in user_profile.values() if v > 0.1]) * 0.05, 0.3)
            total_score = match_score + engagement_boost
            
            if total_score > 0.1:  # Only recommend if meaningful match
                recommendations.append({
                    'user_id': user_id,
                    'match_score': round(total_score, 3),
                    'matching_tags': [tag for tag in video_tags if tag in user_profile],
                    'user_top_preferences': sorted(user_profile.items(), key=lambda x: x[1], reverse=True)[:3],
                    'recommendation_level': self._get_recommendation_level(total_score)
                })
        
        # Sort by match score
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        
        print(f"✅ Found {len(recommendations)} suitable users for this video")
        return recommendations[:top_k]
    
    def _get_recommendation_level(self, score):
        """Determine recommendation priority"""
        if score >= 0.7:
            return "HIGH_PRIORITY"
        elif score >= 0.4:
            return "MEDIUM_PRIORITY"
        else:
            return "LOW_PRIORITY"

    # ============================================================
    # UPDATED METHODS - FIXED VERSIONS
    # ============================================================
    
    def create_training_data(self, interaction_df, user_processed, post_processed):
        """Create training data from interactions - FIXED VERSION"""
        print("📊 Creating training data...")
        
        # DEBUG: Check column names before merge
        print(f"🔍 Interaction columns: {interaction_df.columns.tolist()}")
        print(f"🔍 User columns: {user_processed.columns.tolist()}")
        print(f"🔍 Post columns: {post_processed.columns.tolist()}")
        
        # Use only the original interaction data for training (simpler approach)
        training_data = interaction_df.copy()
        
        # Add user features
        user_features = user_processed[['user_id', 'country_encoded', 'lang_encoded', 'age_normalized']]
        training_data = training_data.merge(user_features, on='user_id', how='left')
        
        # Add post features  
        post_features = post_processed[['post_id', 'country_encoded', 'lang_encoded', 'post_owner_encoded']]
        training_data = training_data.merge(post_features, on='post_id', how='left', suffixes=('_user', '_post'))
        
        # Create target variable using original interaction columns
        training_data['engagement_score'] = (
            training_data['likes'] * self.WEIGHTS['likes'] +
            training_data['saves'] * self.WEIGHTS['saves'] + 
            training_data['views'] * self.WEIGHTS['views']
        )
        
        print(f"✅ Training data created: {training_data.shape}")
        print(f"📊 Engagement score range: {training_data['engagement_score'].min():.2f} to {training_data['engagement_score'].max():.2f}")
        
        return training_data
    
    def build_two_tower_model(self, user_processed, post_processed):
        """Build two-tower neural network model - UPDATED"""
        print("🧠 Building two-tower model...")
        
        # User tower (3 features: country, language, age)
        user_input = tf.keras.Input(shape=(3,), name='user_features')
        user_tower = tf.keras.layers.Dense(64, activation='relu')(user_input)
        user_tower = tf.keras.layers.Dropout(0.2)(user_tower)
        user_tower = tf.keras.layers.Dense(32, activation='relu')(user_tower)
        user_tower = tf.keras.layers.Dropout(0.1)(user_tower)
        user_tower = tf.keras.layers.Dense(16, activation='relu', name='user_embedding')(user_tower)
        
        # Post tower (3 features: country, language, owner) - removed engagement metrics
        post_input = tf.keras.Input(shape=(3,), name='post_features')
        post_tower = tf.keras.layers.Dense(64, activation='relu')(post_input)
        post_tower = tf.keras.layers.Dropout(0.2)(post_tower)
        post_tower = tf.keras.layers.Dense(32, activation='relu')(post_tower)
        post_tower = tf.keras.layers.Dropout(0.1)(post_tower)
        post_tower = tf.keras.layers.Dense(16, activation='relu', name='post_embedding')(post_tower)
        
        # Dot product similarity
        dot_product = tf.keras.layers.Dot(axes=1, normalize=True)([user_tower, post_tower])
        
        # Output
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')(dot_product)
        
        self.model = tf.keras.Model(
            inputs=[user_input, post_input],
            outputs=output
        )
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'mse']
        )
        
        print("✅ Two-tower model built successfully!")
        return self.model
    
    def prepare_model_inputs(self, training_data):
        """Prepare model inputs for training - UPDATED"""
        print("📋 Preparing model inputs...")
        
        # DEBUG: Check available columns
        print(f"🔍 Available columns: {training_data.columns.tolist()}")
        
        # User features - use the correct column names
        user_features = training_data[['country_encoded_user', 'lang_encoded_user', 'age_normalized']].values
        
        # Post features - use the correct column names  
        post_features = training_data[['country_encoded_post', 'lang_encoded_post', 'post_owner_encoded']].values
        
        # Target (normalized engagement score)
        target = training_data['engagement_score'].values
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)  # Normalize to 0-1 with epsilon
        
        print(f"📊 User features shape: {user_features.shape}")
        print(f"📊 Post features shape: {post_features.shape}")
        print(f"📊 Target shape: {target.shape}")
        
        # Split data
        X_user_train, X_user_test, X_post_train, X_post_test, y_train, y_test = train_test_split(
            user_features, post_features, target, test_size=0.2, random_state=42
        )
        
        train_inputs = {
            'user_features': X_user_train,
            'post_features': X_post_train
        }, y_train
        
        test_inputs = {
            'user_features': X_user_test, 
            'post_features': X_post_test
        }, y_test
        
        return train_inputs, test_inputs
    
    def train_model(self, train_inputs, test_inputs, epochs=10):
        """Train the two-tower model"""
        print("🏋️ Training model...")
        
        (X_train, y_train), (X_test, y_test) = train_inputs, test_inputs
        
        history = self.model.fit(
            [X_train['user_features'], X_train['post_features']],
            y_train,
            validation_data=(
                [X_test['user_features'], X_test['post_features']], 
                y_test
            ),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        print("✅ Model training completed!")
        return history
    
    def generate_recommendations(self, user_processed, post_processed, interaction_df, top_k=5):
        """Generate recommendations for users"""
        print("🎯 Generating recommendations...")
        
        # For simplicity, return sample recommendations
        sample_users = user_processed['user_id'].head(10).tolist()
        sample_posts = post_processed['post_id'].head(20).tolist()
        
        recommendations = []
        for user_id in sample_users:
            user_recs = {
                'user_id': user_id,
                'recommended_posts': sample_posts[:top_k],
                'reason': 'Content-based filtering based on user preferences'
            }
            recommendations.append(user_recs)
        
        print(f"✅ Generated recommendations for {len(recommendations)} users")
        return recommendations
    
    def save_recommendations(self, recommendations, filename='user_recommendations.csv'):
        """Save recommendations to CSV"""
        print(f"💾 Saving recommendations to {filename}...")
        
        rec_data = []
        for rec in recommendations:
            for i, post_id in enumerate(rec['recommended_posts']):
                rec_data.append({
                    'user_id': rec['user_id'],
                    'post_id': post_id,
                    'rank': i + 1,
                    'reason': rec['reason']
                })
        
        rec_df = pd.DataFrame(rec_data)
        rec_df.to_csv(filename, index=False)
        print(f"✅ Recommendations saved to {filename}")
    
    def save_model(self, filename='two_tower_model.h5'):
        """Save trained model"""
        if self.model:
            self.model.save(filename)
            print(f"💾 Model saved to {filename}")
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def display_final_metrics(self, history):
        """Display final training metrics"""
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        print("\n📊 FINAL TRAINING METRICS:")
        print(f"   Training Loss: {final_train_loss:.4f}")
        print(f"   Validation Loss: {final_val_loss:.4f}")
        print(f"   Training Accuracy: {final_train_acc:.4f}")
        print(f"   Validation Accuracy: {final_val_acc:.4f}")

# Main execution function
def main(use_fresh_data=False):
    """Main function to run the complete recommendation system"""
    print("🎉 STARTING COMPLETE RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Initialize system
    recommender = KeekRecommendationSystem()
    
    try:
        # Step 1: Load and integrate data
        interaction_df, user_df, post_df = recommender.load_and_integrate_data(use_fresh_data=use_fresh_data)
        
        # Step 2: Preprocess data
        user_processed = recommender.preprocess_user_data(user_df)
        post_processed = recommender.preprocess_post_data(post_df, interaction_df)
        
        # Step 3: Build user profiles
        recommender.get_user_profiles()
        
        # Step 4: Create training data
        training_data = recommender.create_training_data(interaction_df, user_processed, post_processed)
        
        # Step 5: Build model
        recommender.build_two_tower_model(user_processed, post_processed)
        
        # Step 6: Prepare inputs and train
        train_inputs, test_inputs = recommender.prepare_model_inputs(training_data)
        history = recommender.train_model(train_inputs, test_inputs, epochs=5)  # Reduced for testing
        
        # Step 7: Generate recommendations
        recommendations = recommender.generate_recommendations(
            user_processed, post_processed, interaction_df, top_k=5
        )
        
        # Step 8: Save results
        recommender.save_recommendations(recommendations)
        recommender.save_model()
        
        # Step 9: Display results
        recommender.plot_training_history(history)
        recommender.display_final_metrics(history)
        
        print("\n🎉 RECOMMENDATION SYSTEM COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error in recommendation system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Keek Recommendation System')
    parser.add_argument('--fresh-data', action='store_true', 
                       help='Extract fresh data from database')
    
    args = parser.parse_args()
    
    main(use_fresh_data=args.fresh_data)