import os
import shutil
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from datetime import datetime

# Import our services
from recommender_system import KeekRecommendationSystem
from vector_search_service import VectorSearchService
from redis_service import RedisService

# Try to import VideoProcessingService
try:
    from video_processing_service import VideoProcessingService
    HAS_VIDEO_PROC = True
except ImportError:
    HAS_VIDEO_PROC = False

app = FastAPI(title="Keek Recommendation API", version="1.0")

# Global instances
recommender = None
vector_service = None
redis_service = None
video_service = None

@app.on_event("startup")
async def startup_event():
    global recommender, vector_service, redis_service, video_service
    
    print("🚀 Starting up Recommendation API...")
    
    # 1. Initialize Recommender System
    recommender = KeekRecommendationSystem()
    model_path = 'two_tower_model_base.h5'
    artifacts_path = 'model_artifacts_base.pkl'
    
    if os.path.exists(model_path) and os.path.exists(artifacts_path):
        recommender.load_artifacts(model_path, artifacts_path)
        print("✅ Model loaded.")
    else:
        print("⚠️ Model artifacts not found. API will fail on inference.")

    # 2. Initialize Vector Search
    vector_service = VectorSearchService()
    if vector_service.load_index():
        print(f"✅ Vector Index loaded ({vector_service.index.ntotal} users).")
    else:
        print("⚠️ Vector Index not found.")

    # 3. Initialize Redis
    redis_service = RedisService()
    
    # 4. Initialize Video Processor
    if HAS_VIDEO_PROC:
        video_service = VideoProcessingService()
        print("✅ Video Processing Service ready.")
    else:
        print("⚠️ Video Processing Service not available (using mock mode).")

@app.get("/")
def read_root():
    return {"status": "online", "service": "Keek Recommendation Engine"}

@app.post("/recommend/{user_id}")
async def recommend_videos(user_id: int, files: List[UploadFile] = File(...)):
    """
    Upload new videos and get a personalized ranking for the specified user.
    """
    if not recommender or not vector_service:
        raise HTTPException(status_code=503, detail="System not initialized")

    # 1. Save Uploaded Files
    temp_dir = f"temp_uploads_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    saved_paths = []
    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)
        
        print(f"📥 Received {len(saved_paths)} videos for User {user_id}")

        # 2. Process Videos (Get Embeddings)
        # We need to turn these files into embeddings.
        # If we have the video processor, we use it. Else we mock.
        
        video_data = []
        if HAS_VIDEO_PROC and video_service:
            # Process real videos
            processed_videos = video_service.process_folder_videos(temp_dir)
            video_data = processed_videos
        else:
            # Mock processing
            for path in saved_paths:
                video_data.append({
                    'video_path': path,
                    'duration_sec': 60,
                    'predicted_tag_1': 'unknown',
                    'predicted_tag_2': 'unknown'
                })
        
        if not video_data:
            raise HTTPException(status_code=400, detail="No valid videos processed")

        # Create DataFrame for Model
        new_posts_processed = pd.DataFrame({
            'video_path': [v['video_path'] for v in video_data],
            'country_encoded': 0, # Default
            'lang_encoded': 0,    # Default
            'post_owner_encoded': np.random.randint(0, 1000, size=len(video_data))
        })

        # Generate Embeddings
        post_model = recommender.get_post_embedding_model()
        post_feats = new_posts_processed[['country_encoded', 'lang_encoded', 'post_owner_encoded']].values.astype('float32')
        video_embeddings = post_model.predict(post_feats, verbose=0)
        
        # Normalize
        video_norms = np.linalg.norm(video_embeddings, axis=1, keepdims=True)
        video_embeddings = video_embeddings / (video_norms + 1e-10)

        # 3. Get User Embedding
        # Find the user in our index
        # vector_service.user_ids is a list where index i -> user_id
        try:
            # Find index of user_id
            # Note: This is slow for large lists. In production, use a dict map.
            # For prototype (50k), it's okay-ish, but let's optimize if possible.
            # vector_service.user_ids is a numpy array or list.
            if isinstance(vector_service.user_ids, np.ndarray):
                user_idx = np.where(vector_service.user_ids == user_id)[0]
                if len(user_idx) == 0:
                    raise ValueError("User not found")
                user_idx = user_idx[0]
            else:
                user_idx = vector_service.user_ids.index(user_id)
                
            user_embedding = vector_service.index.reconstruct(int(user_idx)).reshape(1, -1)
            
        except (ValueError, IndexError):
            # User not found in Vector DB (New User?)
            # Fallback: Use average user embedding or random
            print(f"⚠️ User {user_id} not found in index. Using neutral embedding.")
            user_embedding = np.zeros((1, 64), dtype='float32') # Or load a default

        # 4. Rank Videos
        # Dot product: (1, 64) x (N, 64).T -> (1, N)
        scores = np.dot(user_embedding, video_embeddings.T).flatten()
        
        # Sort indices desc
        ranked_indices = scores.argsort()[::-1]
        
        recommendations = []
        for idx in ranked_indices:
            recommendations.append({
                'video_name': os.path.basename(new_posts_processed.iloc[idx]['video_path']),
                'score': float(scores[idx]),
                'type': 'new_upload'
            })

        # 5. Discovery Mix (Personalized + Trending)
        # Strategy: 70% Personalized (Top 7), 30% Trending (Random 3 from popular)
        # This ensures users see new popular content even if it doesn't perfectly match their profile.
        
        final_feed = []
        
        # A. Add Top Personalized
        personalized_count = 7
        final_feed.extend(recommendations[:personalized_count])
        
        # B. Add Trending/Popular (Real from Redis)
        trending_count = 3
        trending_pool = []
        
        if redis_service.enabled:
            # Fetch top 10 trending from Redis
            real_trending = redis_service.get_trending_videos(limit=10)
            for vid_id, score in real_trending:
                trending_pool.append({
                    'video_name': vid_id,
                    'score': score,
                    'type': 'trending_discovery'
                })
        
        # If Redis is empty or disabled, fallback to mock
        if not trending_pool:
             trending_pool = [
                {'video_name': f"trending_viral_{i}.mp4", 'score': 0.95, 'type': 'trending_discovery'}
                for i in range(1, 20)
            ]
            
        # Pick random 3 from the pool (Shuffle to keep it fresh)
        import random
        random.shuffle(trending_pool)
        final_feed.extend(trending_pool[:trending_count])
        
        # Pick random trending videos
        import random
        selected_trending = random.sample(trending_pool, min(len(trending_pool), trending_count))
        
        final_feed.extend(selected_trending)
        
        # C. Sparse Data Fallback (if we still don't have 10 items)
        # E.g. if we only had 2 personalized videos, we have 2 + 3 = 5 items. Need 5 more.
        target_total = 10
        if len(final_feed) < target_total:
            needed = target_total - len(final_feed)
            print(f"⚠️ Feed has {len(final_feed)} items. Filling {needed} slots with more Trending.")
            
            # Add more trending (excluding ones already picked if possible, but simple append here)
            extra_trending = [
                {'video_name': f"popular_fallback_{i}.mp4", 'score': 0.88, 'type': 'trending_fallback'} 
                for i in range(needed)
            ]
            final_feed.extend(extra_trending)

        # Shuffle? Usually we keep personalized at top, or interleave.
        # Let's keep personalized at top for relevance, then trending.
        # Or shuffle to make it feel "organic". User asked for "randomly".
        # Let's shuffle the trending items into the bottom half or just append.
        # For now, simple append is fine, but let's return the list.

        return {
            "user_id": user_id,
            "feed": final_feed[:target_total]
        }

    finally:
        # Cleanup temp files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up {temp_dir}")

@app.post("/interact")
async def record_interaction(
    user_id: str = Form(...),
    video_id: str = Form(...),
    action: str = Form(...) # 'like', 'view', 'save'
):
    """
    Record a user interaction (Like/View) to update Real-Time Trending.
    """
    try:
        # 1. Determine Score Delta
        score_delta = 0
        if action == 'like':
            score_delta = 1.0
        elif action == 'save':
            score_delta = 2.0
        elif action == 'view':
            score_delta = 0.1
            
        # 2. Update Redis (Real-Time Trending)
        if redis_service.enabled and score_delta > 0:
            redis_service.update_video_score(video_id, score_delta)
        
        # 3. Persist to CSV (Long-Term Memory)
        # We append to a log file. The Hourly Job can later merge this into the main dataset.
        log_file = "data/interaction_log.csv"
        os.makedirs("data", exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        # Check if file exists to write header
        file_exists = os.path.exists(log_file)
        
        with open(log_file, "a") as f:
            if not file_exists:
                f.write("user_id,post_id,action,timestamp\n")
            f.write(f"{user_id},{video_id},{action},{timestamp}\n")

        return {
            "status": "success", 
            "message": f"Recorded {action} for {video_id}", 
            "new_score_delta": score_delta,
            "persisted": True
        }

    except Exception as e:
        print(f"❌ Error recording interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)