# -*- coding: utf-8 -*-
"""recommender_system.py

Reusable Recommendation System Module
Contains the core logic for:
1. Data Integration (Video + Interactions)
2. User Profiling
3. Two-Tower Model Building & Training
4. Recommendation Generation
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
import pickle
import joblib

warnings.filterwarnings('ignore')

# ============================================================
# VIDEO PROCESSING INTEGRATION
# ============================================================
try:
    from video_processing_service import VideoProcessingService, RealTimeCompatibleVideoProcessor
    HAS_VIDEO_PROCESSING = True
except ImportError as e:
    print(f"⚠️ VideoProcessingService not available: {e}")
    HAS_VIDEO_PROCESSING = False

# ============================================================
# DATABASE EXTRACTION INTEGRATION
# ============================================================
try:
    from production_data_extractor import KeekDataExtractor
    HAS_DATABASE_EXTRACTION = True
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
        Step 1: Process videos and merge with interaction data
        If videos are missing, randomize tags for each post_id.
        """
        print("🎬 STEP 1: VIDEO PROCESSING AND DATA INTEGRATION")
        print("=" * 60)
        
        # Load original interaction data
        print("📊 Loading interaction data...")
        interaction_df = pd.read_csv(interaction_csv_path)
        print(f"✅ Loaded {len(interaction_df)} interactions")
        
        # Get unique post IDs
        unique_post_ids = interaction_df['post_id'].unique()
        print(f"   Found {len(unique_post_ids)} unique posts")
        
        # Randomize tags for all posts (since we don't have actual videos for 2.9M posts)
        print("🎲 Randomizing video tags for all posts...")
        
        # We need the TAGS list. It's defined in KeekRecommendationSystem but not here.
        # Let's define it here or pass it. For now, hardcode or import.
        TAGS = [
            "sports", "gaming", "cooking", "traveling", "entertainment", "music",
            "education", "news", "vlog", "review", "fashion", "technology",
            "fitness", "food", "animals", "nature", "comedy"
        ]
        
        video_data = []
        import random
        
        for post_id in unique_post_ids:
            # Pick 1 or 2 random tags
            num_tags = random.choice([1, 2])
            selected_tags = random.sample(TAGS, num_tags)
            
            tag1 = selected_tags[0]
            tag2 = selected_tags[1] if num_tags > 1 else tag1
            
            # Random duration between 15s and 60s
            duration = round(random.uniform(15.0, 60.0), 2)
            
            video_data.append({
                'post_id': post_id,
                'video_duration_sec': duration,
                'video_predicted_tag_1': tag1,
                'video_predicted_tag_2': tag2
            })
            
        video_df = pd.DataFrame(video_data)
        print(f"✅ Generated random tags for {len(video_df)} posts")
        
        # Merge with interaction data
        print("🔄 Merging video data with interaction table...")
        merged_df = pd.merge(
            interaction_df,
            video_df,
            on='post_id',
            how='left'
        )
        
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
        
        # Encoders and Scalers
        self.country_encoder = LabelEncoder()
        self.lang_encoder = LabelEncoder()
        self.owner_encoder = LabelEncoder()
        self.age_scaler = StandardScaler()
        self.engagement_scaler = StandardScaler()
        
    def extract_and_load_data(self, use_existing=True):
        """Extract data from database or load from CSV"""
        data_dir = "production_data" # Updated to point to production_data
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
                user_df.to_csv(os.path.join(data_dir, 'user_table.csv'), index=False)
                post_df.to_csv(os.path.join(data_dir, 'post_table.csv'), index=False)
                
                self.interaction_df = interaction_df
                print("✅ Fresh data extracted and saved successfully!")
                return interaction_df, user_df, post_df
                
            except ImportError:
                print("❌ Data extractor not available, using existing CSV files")
                return self.load_production_data()
    
    def load_production_data(self):
        """Load data from production CSV files"""
        print("📊 Loading production data from CSV files...")
        data_dir = "production_data"
        
        try:
            # Load interaction data
            interaction_path = os.path.join(data_dir, 'interaction_table.csv')
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
            # NOTE: For production scale, we might skip video processing for all 2.9M interactions
            # and rely on pre-computed tags or just use metadata.
            # For now, we'll keep it but be mindful of performance.
            
            # Step 2: Process videos and merge with interaction data
            # We force processing here to ensure we get the full dataset with randomized tags
            # instead of loading potentially stale/truncated cached data.
            
            # Define path to raw interaction CSV for processing
            interaction_csv_path = "production_data/interaction_table.csv"
            
            # Call the integrator to randomize tags and merge
            self.interaction_df = self.video_integrator.process_videos_and_merge(interaction_csv_path)
            
            return self.interaction_df, user_df, post_df

            return self.interaction_df, user_df, post_df
            
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            raise
    
    def preprocess_user_data(self, user_df, is_training=True):
        """Preprocess user demographic data"""
        print("👤 Preprocessing user data...")
        
        user_processed = user_df.copy()
        
        # Country encoding
        if is_training:
            user_processed['country_encoded'] = self.country_encoder.fit_transform(
                user_processed['country'].fillna('unknown')
            )
        else:
            # Handle unseen labels
            user_processed['country'] = user_processed['country'].fillna('unknown')
            # Use a safe transform that handles unseen labels (e.g. maps to 'unknown' or similar)
            # For simplicity here, we'll use a try-except or map to a default if not found
            # A robust way is to use a custom transformer or check classes
            known_countries = set(self.country_encoder.classes_)
            user_processed['country'] = user_processed['country'].apply(lambda x: x if x in known_countries else 'unknown')
            # If 'unknown' wasn't in training, we might have an issue. 
            # Ideally 'unknown' should be in training classes.
            if 'unknown' not in known_countries:
                 # Force fit 'unknown' if not present (hacky for inference but needed)
                 # Better: fit on training data that includes 'unknown'
                 pass
            
            try:
                user_processed['country_encoded'] = self.country_encoder.transform(user_processed['country'])
            except ValueError:
                # Fallback for completely new labels
                user_processed['country_encoded'] = 0 # Default to 0

        
        # Language encoding
        def extract_primary_language(lang_string):
            try:
                if isinstance(lang_string, str) and lang_string.strip():
                    clean_string = lang_string.replace('"', '').replace('[', '').replace(']', '')
                    languages = [lang.strip() for lang in clean_string.split(',') if lang.strip()]
                    return languages[0] if languages else 'en'
                return 'en'
            except Exception as e:
                # print(f"⚠️ Language parsing error for {lang_string}: {e}")
                return 'en'
        
        user_processed['primary_language'] = user_processed['supported_language'].apply(extract_primary_language)
        
        if is_training:
            user_processed['lang_encoded'] = self.lang_encoder.fit_transform(
                user_processed['primary_language'].fillna('en')
            )
        else:
            known_langs = set(self.lang_encoder.classes_)
            user_processed['primary_language'] = user_processed['primary_language'].apply(lambda x: x if x in known_langs else 'en')
            user_processed['lang_encoded'] = self.lang_encoder.transform(user_processed['primary_language'])
        
        # Age normalization
        if 'age' in user_processed.columns:
            if is_training:
                self.age_scaler.fit(user_processed[['age']].fillna(user_processed['age'].mean()))
            
            user_processed['age_normalized'] = self.age_scaler.transform(user_processed[['age']].fillna(user_processed['age'].mean()))
        else:
            user_processed['age_normalized'] = 0
        
        # Create user_id column for consistency
        if 'user_id' not in user_processed.columns and 'id' in user_processed.columns:
            user_processed['user_id'] = user_processed['id']
        
        self.user_processed = user_processed
        
        print("✅ User data preprocessed successfully!")
        return user_processed
    
    def preprocess_post_data(self, post_df, interaction_df, is_training=True):
        """Preprocess post/video data"""
        print("🎬 Preprocessing post data...")
        
        post_processed = post_df.copy()
        
        # Country and language encoding
        if is_training:
            # Re-use country encoder if possible, or fit new one? 
            # Usually better to share if domain is same, but let's fit separately for posts to be safe
            # Or use the same one. For now, we'll fit a new one for posts or use the same logic.
            # Let's use a separate encoder for posts to avoid conflicts if sets differ significantly
            # But wait, we didn't define a separate one in __init__. 
            # Let's define it now or reuse.
            # Actually, reusing might be better if we want shared embedding space, but here they are separate inputs.
            pass

        # We need encoders for posts. Let's add them to __init__ or just use local ones if not saving.
        # But we NEED to save them for inference.
        if not hasattr(self, 'post_country_encoder'):
             self.post_country_encoder = LabelEncoder()
        if not hasattr(self, 'post_lang_encoder'):
             self.post_lang_encoder = LabelEncoder()
             
        if is_training:
            post_processed['country_encoded'] = self.post_country_encoder.fit_transform(
                post_processed['country'].fillna('unknown')
            )
            post_processed['lang_encoded'] = self.post_lang_encoder.fit_transform(
                post_processed['lang'].fillna('en')
            )
            post_processed['post_owner_encoded'] = self.owner_encoder.fit_transform(
                post_processed['post_owner_id']
            )
        else:
            # Inference logic
            known_countries = set(self.post_country_encoder.classes_)
            post_processed['country'] = post_processed['country'].fillna('unknown').apply(lambda x: x if x in known_countries else 'unknown')
            # Handle case where 'unknown' is not in classes
            if 'unknown' not in known_countries and len(known_countries) > 0:
                 # Map to first class as fallback
                 fallback = list(known_countries)[0]
                 post_processed['country'] = post_processed['country'].replace('unknown', fallback)
            
            post_processed['country_encoded'] = self.post_country_encoder.transform(post_processed['country'])
            
            known_langs = set(self.post_lang_encoder.classes_)
            post_processed['lang'] = post_processed['lang'].fillna('en').apply(lambda x: x if x in known_langs else 'en')
            post_processed['lang_encoded'] = self.post_lang_encoder.transform(post_processed['lang'])
            
            # Owner ID is tricky for new owners. 
            # We can map unknown owners to a special token or 0.
            known_owners = set(self.owner_encoder.classes_)
            # For now, map unknown to existing or handle error.
            # A simple hack: use a default owner or mod.
            # Better: Use a hash bucket or similar. 
            # Here: We'll just clip to known range or use 0.
            post_processed['post_owner_encoded'] = post_processed['post_owner_id'].apply(lambda x: self.owner_encoder.transform([x])[0] if x in known_owners else 0)

        
        # Calculate post engagement metrics
        # For inference, we might not have interaction history for new posts.
        # So we should handle that.
        if interaction_df is not None:
            post_engagement = interaction_df.groupby('post_id').agg({
                'likes': 'sum', 'views': 'sum', 'saves': 'sum'
            }).reset_index()
            
            # Merge with post data
            post_processed = post_processed.merge(post_engagement, on='post_id', how='left')
        else:
            # No interactions (e.g. new posts)
            post_processed['likes'] = 0
            post_processed['views'] = 0
            post_processed['saves'] = 0
        
        # Fill NaN values and normalize
        engagement_cols = ['likes', 'views', 'saves']
        post_processed[engagement_cols] = post_processed[engagement_cols].fillna(0)
        
        if is_training:
            self.engagement_scaler.fit(post_processed[engagement_cols])
            
        post_processed[engagement_cols] = self.engagement_scaler.transform(post_processed[engagement_cols])
        
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
    
    def build_user_profiles(self, interaction_df):
        """Build dynamic user profiles based on interaction history"""
        print("🔍 Building user profiles from interactions...")
        
        user_profiles = {}
        
        # Optimization: Use vectorization instead of iterrows if possible, 
        # but for complex logic iterrows is okay for now.
        # Given 2.9M rows, iterrows will be SLOW.
        # TODO: Optimize this for production.
        
        # For now, we'll stick to the original logic but warn about speed.
        print("⚠️ Note: Building profiles iteratively. This may take time for large datasets.")
        
        # Simplified profile building for speed:
        # Group by user and aggregate tags?
        # We need video tags for this.
        
        # If we don't have video tags in interaction_df, we can't build content-based profiles easily.
        # We'll assume interaction_df has 'video_predicted_tag_1' etc. if integrated.
        
        if 'video_predicted_tag_1' not in interaction_df.columns:
             print("⚠️ No video tags found in interactions. Skipping content-based profile building.")
             return {}

        for idx, row in interaction_df.iterrows():
            user_id = row['user_id']
            
            # Calculate engagement score
            engagement_score = (
                row['likes'] * self.WEIGHTS['likes'] +
                row['saves'] * self.WEIGHTS['saves'] +
                row['views'] * self.WEIGHTS['views']
            )
            
            post_tags = [row.get('video_predicted_tag_1', 'unknown'), 
                        row.get('video_predicted_tag_2', 'unknown')]
            learning_rate = 0.1
            
            if user_id not in user_profiles:
                user_profiles[user_id] = {category: 1.0/len(self.TAGS) for category in self.TAGS}
            
            current_weights = user_profiles[user_id].copy()
            for tag in post_tags:
                if tag in current_weights:
                    current_weights[tag] += engagement_score * learning_rate
            
            total_weight = sum(current_weights.values())
            if total_weight > 0:
                user_profiles[user_id] = {category: weight / total_weight for category, weight in current_weights.items()}
            
            if idx % 10000 == 0 and idx > 0:
                print(f"Processed {idx}/{len(interaction_df)} interactions")
        
        self.user_profiles = user_profiles
        print(f"✅ User profiles built for {len(user_profiles)} users!")
        return user_profiles

    def create_training_data(self, interaction_df, user_processed, post_processed):
        """Create training data from interactions"""
        print("📊 Creating training data...")
        
        training_data = interaction_df.copy()
        
        # Add user features
        print(f"   Merging user features... Interaction rows: {len(training_data)}, User rows: {len(user_processed)}")
        # Check dtypes
        # print(f"   User ID types - Interaction: {training_data['user_id'].dtype}, User: {user_processed['user_id'].dtype}")
        
        user_features = user_processed[['user_id', 'country_encoded', 'lang_encoded', 'age_normalized']]
        training_data = training_data.merge(user_features, on='user_id', how='left')
        print(f"   After user merge: {len(training_data)} rows")
        
        # Add post features  
        print(f"   Merging post features... Post rows: {len(post_processed)}")
        post_features = post_processed[['post_id', 'country_encoded', 'lang_encoded', 'post_owner_encoded']]
        training_data = training_data.merge(post_features, on='post_id', how='left', suffixes=('_user', '_post'))
        print(f"   After post merge: {len(training_data)} rows")
        
        # Create target variable
        training_data['engagement_score'] = (
            training_data['likes'] * self.WEIGHTS['likes'] +
            training_data['saves'] * self.WEIGHTS['saves'] + 
            training_data['views'] * self.WEIGHTS['views']
        )
        
        print(f"✅ Training data created: {training_data.shape}")
        
        # Check for NaNs
        if training_data.isnull().any().any():
            print("⚠️ Found NaNs in training data. Filling with 0.")
            training_data = training_data.fillna(0)
            
        return training_data
    
    def build_two_tower_model(self, user_processed, post_processed):
        """Build two-tower neural network model"""
        print("🧠 Building two-tower model...")
        
        # User tower
        user_input = tf.keras.Input(shape=(3,), name='user_features')
        user_bn = tf.keras.layers.BatchNormalization()(user_input)
        user_tower = tf.keras.layers.Dense(64, activation='relu')(user_bn)
        user_tower = tf.keras.layers.Dropout(0.2)(user_tower)
        user_tower = tf.keras.layers.Dense(32, activation='relu')(user_tower)
        user_tower = tf.keras.layers.Dense(64, activation='relu', name='user_embedding')(user_tower)
        
        # Post tower
        post_input = tf.keras.Input(shape=(3,), name='post_features')
        post_bn = tf.keras.layers.BatchNormalization()(post_input)
        post_tower = tf.keras.layers.Dense(128, activation='relu')(post_bn)
        post_tower = tf.keras.layers.Dropout(0.2)(post_tower)
        post_tower = tf.keras.layers.Dense(64, activation='relu')(post_tower)
        post_tower = tf.keras.layers.Dense(64, activation='relu', name='post_embedding')(post_tower)
        
        # Dot product similarity
        dot_product = tf.keras.layers.Dot(axes=1, normalize=True)([user_tower, post_tower])
        
        # Output
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')(dot_product)
        
        self.model = tf.keras.Model(
            inputs=[user_input, post_input],
            outputs=output
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy', 'mse']
        )
        
        print("✅ Two-tower model built successfully!")
        return self.model

    def get_user_embedding_model(self):
        """Extract the user tower as a separate model"""
        if not self.model:
            raise ValueError("Model not built yet!")
            
        user_input = self.model.get_layer('user_features').input
        user_embedding = self.model.get_layer('user_embedding').output
        return tf.keras.Model(user_input, user_embedding)

    def get_post_embedding_model(self):
        """Extract the post tower as a separate model"""
        if not self.model:
            raise ValueError("Model not built yet!")
            
        post_input = self.model.get_layer('post_features').input
        post_embedding = self.model.get_layer('post_embedding').output
        return tf.keras.Model(post_input, post_embedding)
    
    def prepare_model_inputs(self, training_data):
        """Prepare model inputs for training"""
        print("📋 Preparing model inputs...")
        
        # User features
        user_features = training_data[['country_encoded_user', 'lang_encoded_user', 'age_normalized']].values
        
        # Post features
        post_features = training_data[['country_encoded_post', 'lang_encoded_post', 'post_owner_encoded']].values
        
        # Target
        target = training_data['engagement_score'].values
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
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
    
    def train_model(self, train_inputs, test_inputs, epochs=5):
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

    def save_artifacts(self, model_path='two_tower_model.h5', artifacts_path='model_artifacts.pkl'):
        """Save model and preprocessing artifacts"""
        print("💾 Saving model and artifacts...")
        
        # Save Keras model
        if self.model:
            self.model.save(model_path)
            print(f"   Model saved to {model_path}")
            
        # Save artifacts (encoders, scalers)
        artifacts = {
            'country_encoder': self.country_encoder,
            'lang_encoder': self.lang_encoder,
            'owner_encoder': self.owner_encoder,
            'post_country_encoder': getattr(self, 'post_country_encoder', None),
            'post_lang_encoder': getattr(self, 'post_lang_encoder', None),
            'age_scaler': self.age_scaler,
            'engagement_scaler': self.engagement_scaler,
            'user_profiles': self.user_profiles
        }
        
        with open(artifacts_path, 'wb') as f:
            pickle.dump(artifacts, f)
        print(f"   Artifacts saved to {artifacts_path}")

    def load_artifacts(self, model_path='two_tower_model.h5', artifacts_path='model_artifacts.pkl'):
        """Load model and preprocessing artifacts"""
        print("Loading model and artifacts...")
        
        # Load Keras model
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"   Model loaded from {model_path}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
            
        # Load artifacts
        if os.path.exists(artifacts_path):
            with open(artifacts_path, 'rb') as f:
                artifacts = pickle.load(f)
            
            self.country_encoder = artifacts.get('country_encoder')
            self.lang_encoder = artifacts.get('lang_encoder')
            self.owner_encoder = artifacts.get('owner_encoder')
            self.post_country_encoder = artifacts.get('post_country_encoder')
            self.post_lang_encoder = artifacts.get('post_lang_encoder')
            self.age_scaler = artifacts.get('age_scaler')
            self.engagement_scaler = artifacts.get('engagement_scaler')
            self.user_profiles = artifacts.get('user_profiles', {})
            
            print(f"   Artifacts loaded from {artifacts_path}")
        else:
            print(f"⚠️ Artifacts file not found: {artifacts_path}")
