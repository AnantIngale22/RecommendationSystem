import time
import numpy as np
import pandas as pd
import os
from recommender_system import KeekRecommendationSystem
from vector_search_service import VectorSearchService

def benchmark():
    print("🚀 Starting Real-Time Scoring Benchmark...")
    print("------------------------------------------------")

    # 1. Load Model (One-time cost, usually loaded in memory)
    print("1. Loading Model & Index (Simulating Server Startup)...")
    start_load = time.time()
    
    recommender = KeekRecommendationSystem()
    recommender.load_artifacts('two_tower_model_base.h5', 'model_artifacts_base.pkl')
    
    vector_service = VectorSearchService()
    vector_service.load_index()
    
    # Get User Embedding (Simulate 1 User)
    # In real-time, we fetch this from Redis/FAISS instantly
    user_id = vector_service.user_ids[0]
    user_embedding = vector_service.index.reconstruct(0).reshape(1, -1)
    
    post_model = recommender.get_post_embedding_model()
    
    print(f"✅ Loaded in {time.time() - start_load:.4f}s")
    print("------------------------------------------------")

    # 2. Benchmark Scenarios
    scenarios = [100, 1000, 5000, 10000, 1000000]
    
    for n_videos in scenarios:
        print(f"\n📊 Scenario: Scoring {n_videos} Videos (End-to-End)")
        
        # Step A: Generate Real CSV File (Simulate Data Arrival)
        csv_filename = f"temp_benchmark_{n_videos}.csv"
        # Create random data
        df = pd.DataFrame({
            'video_path': [f"video_{i}.mp4" for i in range(n_videos)],
            'country_encoded': np.random.randint(0, 100, size=n_videos),
            'lang_encoded': np.random.randint(0, 50, size=n_videos),
            'post_owner_encoded': np.random.randint(0, 1000, size=n_videos),
            'tags': [np.random.choice(['funny, viral', 'sports, action', 'news, politics', 'gaming, stream'], 1)[0] for _ in range(n_videos)]
        })
        df.to_csv(csv_filename, index=False)
        
        # Start Timer (End-to-End Latency)
        start_time = time.time()
        
        # Step B: Read CSV (I/O Cost)
        df_loaded = pd.read_csv(csv_filename)
        io_time = time.time()
        
        # Step C: Preprocessing (Dataframe to Numpy)
        post_feats = df_loaded[['country_encoded', 'lang_encoded', 'post_owner_encoded']].values.astype('float32')
        prep_time = time.time()
        
        # Step D: Model Inference (Get Embeddings)
        video_embeddings = post_model.predict(post_feats, verbose=0)
        inference_time = time.time()
        
        # Step E: Normalize
        video_norms = np.linalg.norm(video_embeddings, axis=1, keepdims=True)
        video_embeddings = video_embeddings / (video_norms + 1e-10)
        
        # Step F: Scoring (Dot Product)
        scores = np.dot(user_embedding, video_embeddings.T).flatten()
        
        # Step G: Ranking (Sort)
        top_indices = scores.argsort()[-10:][::-1]
        
        end_time = time.time()
        
        # Cleanup
        # if os.path.exists(csv_filename):
        #     os.remove(csv_filename)
            
        # Calculate Durations
        total_duration = (end_time - start_time) * 1000
        io_duration = (io_time - start_time) * 1000
        prep_duration = (prep_time - io_time) * 1000
        inf_duration = (inference_time - prep_time) * 1000
        rank_duration = (end_time - inference_time) * 1000
        
        print(f"   ⏱️  Total Latency: {total_duration:.2f} ms")
        print(f"       - CSV Read (I/O): {io_duration:.2f} ms")
        print(f"       - Preprocessing:  {prep_duration:.2f} ms")
        print(f"       - Model Inference:{inf_duration:.2f} ms")
        print(f"       - Score & Rank:   {rank_duration:.2f} ms")
        
        if total_duration < 200:
            print("   ✅ Status: EXCELLENT (< 200ms)")
        elif total_duration < 500:
            print("   ⚠️ Status: ACCEPTABLE (< 500ms)")
        else:
            print("   ❌ Status: SLOW (> 500ms)")

    print("\n------------------------------------------------")
    print("💡 CONCLUSION:")
    print("If Video Vectors are CACHED (Pre-computed), latency is just 'Scoring & Ranking'.")
    print("If Videos are NEW (Full Inference), latency includes 'Model Inference'.")

if __name__ == "__main__":
    benchmark()
