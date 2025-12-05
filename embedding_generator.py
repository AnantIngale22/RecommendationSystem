import os
import numpy as np
import pandas as pd
from recommender_system import KeekRecommendationSystem
from vector_search_service import VectorSearchService

def main():
    print("🚀 STARTING EMBEDDING GENERATION JOB")
    print("="*60)
    
    # Initialize system
    recommender = KeekRecommendationSystem()
    
    # Step 1: Load Model
    print("\n📥 Loading trained model...")
    try:
        recommender.load_artifacts(
            model_path='two_tower_model_base.h5',
            artifacts_path='model_artifacts_base.pkl'
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Step 2: Load All Users
    print("\n👥 Loading user data...")
    _, user_df, _ = recommender.load_production_data()
    print(f"   Found {len(user_df)} users.")
    
    # Step 3: Preprocess Users
    print("\n⚙️ Preprocessing users...")
    user_processed = recommender.preprocess_user_data(user_df, is_training=False)
    
    # Step 4: Generate Embeddings
    print("\n🧮 Generating User Embeddings...")
    user_model = recommender.get_user_embedding_model()
    
    # Prepare inputs
    user_features = user_processed[['country_encoded', 'lang_encoded', 'age_normalized']].values.astype('float32')
    
    # Batch prediction to avoid OOM
    batch_size = 1000
    embeddings = []
    
    for i in range(0, len(user_features), batch_size):
        batch = user_features[i:i+batch_size]
        batch_emb = user_model.predict(batch, verbose=0)
        embeddings.append(batch_emb)
        if i % 10000 == 0:
            print(f"   Processed {i}/{len(user_features)} users")
            
    all_embeddings = np.vstack(embeddings)
    print(f"✅ Generated embeddings shape: {all_embeddings.shape}")
    
    # Step 5: Build and Save FAISS Index
    print("\n🏗️ Building Vector Index...")
    vector_service = VectorSearchService()
    
    user_ids = user_processed['user_id'].tolist()
    vector_service.build_index(all_embeddings, user_ids)
    
    # Save
    vector_service.save_index()
    print("\n🎉 EMBEDDING GENERATION COMPLETED!")

if __name__ == "__main__":
    main()
