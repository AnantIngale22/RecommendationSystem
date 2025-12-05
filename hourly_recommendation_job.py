# -*- coding: utf-8 -*-
"""hourly_recommendation_job.py

Hourly Inference Job
Loads the pre-trained model and generates recommendations for NEW data (e.g., new videos).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from recommender_system import KeekRecommendationSystem

def main():
    print("⏰ STARTING HOURLY RECOMMENDATION JOB")
    print("="*60)
    
    try:
        # Initialize system
        recommender = KeekRecommendationSystem()
        
        # Step 1: Load Pre-trained Model and Artifacts
        print("\n📥 STEP 1: Loading pre-trained model...")
        model_path = 'two_tower_model_base.h5'
        artifacts_path = 'model_artifacts_base.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
            print("❌ Pre-trained model or artifacts not found! Please run train_base_model.py first.")
            return

        recommender.load_artifacts(model_path, artifacts_path)
        
        # Step 2: Load "Hourly" Data (New Videos)
        print("\n🎥 STEP 2: Scanning for new content (Hourly Data)...")
        
        new_posts_processed = pd.DataFrame()
        
        # Check for CSV first (User provided "new_data/videos.csv")
        csv_path = "new_data/videos.csv"
        
        if os.path.exists(csv_path):
            print(f"📄 Found CSV data: {csv_path}")
            csv_df = pd.read_csv(csv_path)
            print(f"   Loaded {len(csv_df)} videos from CSV.")
            
            # Create mock features for the model
            # The model needs: country_encoded, lang_encoded, post_owner_encoded
            # We will use default values (e.g., 0) or random values for the prototype
            new_posts_processed = pd.DataFrame({
                'video_path': csv_df['video_name'],
                'country_encoded': 0, # Default
                'lang_encoded': 0,    # Default
                'post_owner_encoded': np.random.randint(0, 1000, size=len(csv_df)) # Random owner
            })
            print(f"✅ Created features for {len(new_posts_processed)} videos from CSV.")
            
        else:
            # Fallback to scanning video files
            try:
                from video_processing_service import VideoProcessingService
                video_service = VideoProcessingService()
                HAS_VIDEO_PROC = True
            except ImportError:
                print("⚠️ VideoProcessingService not available. Using DUMMY data.")
                HAS_VIDEO_PROC = False
                
            new_videos = []
            
            if HAS_VIDEO_PROC:
                # Scan for new videos
                all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
                new_data_dir = next((d for d in all_dirs if d.startswith('new_data')), None)
                
                if not new_data_dir:
                    new_data_dir = "new_videos"
                    if not os.path.exists(new_data_dir):
                        os.makedirs(new_data_dir, exist_ok=True)
                        print(f"   Created {new_data_dir}. Please add videos here.")
                        return

                print(f"   Scanning directory: {new_data_dir}")
                new_videos = video_service.process_folder_videos(new_data_dir)
            else:
                # Create dummy video data for verification
                new_videos = [
                    {'video_path': 'dummy_video_1.mp4', 'duration_sec': 60},
                    {'video_path': 'dummy_video_2.mp4', 'duration_sec': 30}
                ]
                print(f"   Generated {len(new_videos)} dummy videos for testing.")
            
            if not new_videos:
                print("⚠️ No new videos found.")
                return
                
            print(f"✅ Processed {len(new_videos)} videos successfully")
            
            # Preprocess new videos for the model
            new_posts_processed = pd.DataFrame({
                'video_path': [v['video_path'] for v in new_videos],
                'country_encoded': 0,
                'lang_encoded': 0,
                'post_owner_encoded': np.random.randint(0, 1000, size=len(new_videos))
            })

        # Step 3: Generate Recommendations
        print("\n🧠 STEP 3: Generating recommendations...")
        
        if new_posts_processed.empty:
            print("⚠️ No posts to process.")
            return

        # Initialize Vector Search Service
        from vector_search_service import VectorSearchService
        vector_service = VectorSearchService()
        if not vector_service.load_index():
            print("❌ Failed to load vector index. Please run embedding_generator.py first.")
            return

        # Initialize Redis Service
        from redis_service import RedisService
        redis_service = RedisService()

        # ---------------------------------------------------------
        # NEW LOGIC: User-Centric Ranking (Feed Generation)
        # ---------------------------------------------------------
        # 1. Generate Embeddings for ALL New Videos
        print(f"   Generating embeddings for {len(new_posts_processed)} new videos...")
        post_model = recommender.get_post_embedding_model()
        
        # Batch predict for efficiency
        post_feats = new_posts_processed[['country_encoded', 'lang_encoded', 'post_owner_encoded']].values.astype('float32')
        video_embeddings = post_model.predict(post_feats, verbose=0)
        
        # Normalize video embeddings (for Cosine Similarity)
        # Note: FAISS index is already normalized, so we should normalize these too if we were searching index.
        # But here we are doing manual Dot Product against User Embeddings.
        video_norms = np.linalg.norm(video_embeddings, axis=1, keepdims=True)
        video_embeddings = video_embeddings / (video_norms + 1e-10)
        
        # Step 4: Calculate "Recent Popularity" (3-Day Window)
        # We read the interaction log and filter for recent events
        print("\n🔥 STEP 4: Calculating Recent Popularity (Last 3 Days)...")
        interaction_log_path = "data/interaction_log.csv"
        recent_popularity_scores = {}
        
        if os.path.exists(interaction_log_path):
            try:
                log_df = pd.read_csv(interaction_log_path)
                
                # Convert timestamp to datetime
                log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
                
                # Filter: Keep only last 3 days
                cutoff_date = datetime.now() - pd.Timedelta(days=3)
                recent_df = log_df[log_df['timestamp'] > cutoff_date]
                
                print(f"   - Total Interactions: {len(log_df)}")
                print(f"   - Recent Interactions (3 Days): {len(recent_df)}")
                
                if not recent_df.empty:
                    # Calculate scores: Like=1, Save=2, View=0.1
                    # We group by video_id and sum the scores
                    def get_score(action):
                        if action == 'like': return 1.0
                        if action == 'save': return 2.0
                        if action == 'view': return 0.1
                        return 0.0
                    
                    recent_df['score_val'] = recent_df['action'].apply(get_score)
                    
                    # Group by post_id (video_id)
                    pop_stats = recent_df.groupby('post_id')['score_val'].sum()
                    recent_popularity_scores = pop_stats.to_dict()
                    
                    print(f"   - Identified {len(recent_popularity_scores)} Trending Videos.")
            except Exception as e:
                print(f"⚠️ Error processing interaction log: {e}")
        else:
            print("   - No interaction log found. Skipping popularity boost.")

        # Step 5: Generate Recommendations for Users
        # We assume we have a list of users to generate for.
        # For this prototype, we'll generate for a sample of users (or all if small).
        print("\n🚀 STEP 5: Generating User Feeds...")
        
        # Load user IDs from artifacts or generate mock
        # In a real system, we'd query the User DB
        num_users = 52328 # From previous context
        user_ids = [200126921366 + i for i in range(100)] # Simulate 100 users for speed
        
        print(f"   Generating feeds for {len(user_ids)} users...")
        
        recommendations = []
        
        # Batch processing
        user_batch_size = 1000
        
        # Pre-calculate video popularity vector (aligned with video_embeddings)
        # If a video in 'new_posts_processed' is in 'recent_popularity_scores', it gets a boost.
        video_pop_boost = np.zeros(len(new_posts_processed))
        for i, row in new_posts_processed.iterrows():
            vid_name = os.path.basename(row['video_path'])
            # Check if this video is trending
            # Note: In log, we stored 'video_id', here we have 'video_path'. 
            # We assume video_id == video_name for matching.
            if vid_name in recent_popularity_scores:
                video_pop_boost[i] = recent_popularity_scores[vid_name] * 0.05 # Small boost factor
        
        # Normalize boost
        if np.max(video_pop_boost) > 0:
            video_pop_boost = video_pop_boost / np.max(video_pop_boost) # 0 to 1 range
        
        # 2. Load Active Users (Production Scale)
        # In a real app, you might fetch "Users active in last 1 hour" from Redis/DB.
        # Here, we will generate feeds for ALL users in our vector index (50k+).
        all_user_ids = vector_service.user_ids
        total_users = len(all_user_ids)
        print(f"   Generating feeds for ALL {total_users} users...")
        
        recommendations = []
        BATCH_SIZE = 5000 # Process users in batches to manage memory
        
        for start_idx in range(0, total_users, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_users)
            batch_ids = all_user_ids[start_idx:end_idx]
            
            # Reconstruct embeddings for this batch
            # We need the integer indices for FAISS reconstruction
            batch_indices = range(start_idx, end_idx)
            
            # Reconstruct batch: (BATCH_SIZE, 64)
            # Note: FAISS reconstruct_n might be faster if available, but loop is fine for 5k
            batch_user_embeddings = np.zeros((len(batch_ids), 64), dtype='float32')
            for i, idx in enumerate(batch_indices):
                batch_user_embeddings[i] = vector_service.index.reconstruct(idx)
            
            # 3. Calculate Scores (Dot Product)
            # Users (B, 64) x Videos (V, 64).T -> (B, V) scores
            # Result: Score for every user-video pair in this batch
            batch_scores = np.dot(batch_user_embeddings, video_embeddings.T)
            
            # 4. Rank Videos for each user in batch
            for i, user_id in enumerate(batch_ids):
                user_scores = batch_scores[i] # (V,)
                
                # Top 10 for this user
                top_indices = user_scores.argsort()[-10:][::-1]
                
                for vid_idx in top_indices:
                    recommendations.append({
                        'timestamp': datetime.now().isoformat(),
                        'user_id': user_id,
                        'video_path': new_posts_processed.iloc[vid_idx]['video_path'],
                        'score': round(float(user_scores[vid_idx]), 4),
                        'type': 'personalized_feed'
                    })
            
            print(f"   Processed users {start_idx} to {end_idx} ({len(recommendations)} recs so far)")
                
        print(f"✅ Generated {len(recommendations)} feed items for {total_users} users.")
        
        # Step 4: Save Recommendations
        print("\n💾 STEP 4: Saving hourly recommendations...")
        rec_df = pd.DataFrame(recommendations)
        output_file = f"hourly_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        rec_df.to_csv(output_file, index=False)
        
        print(f"✅ Saved {len(rec_df)} recommendations to {output_file}")
        
        # Cache to Redis
        if redis_service.enabled and not rec_df.empty:
            print("   Caching recommendations to Redis...")
            # Group by user
            grouped = rec_df.groupby('user_id')
            for user_id, group in grouped:
                # Convert to list of dicts
                user_recs = group[['video_path', 'score', 'timestamp', 'type']].to_dict('records')
                redis_service.cache_recommendations(user_id, user_recs)
            print(f"✅ Cached recommendations for {len(grouped)} users.")
        print("\n🎉 HOURLY JOB COMPLETED!")

    except Exception as e:
        print(f"❌ Error in hourly job: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
