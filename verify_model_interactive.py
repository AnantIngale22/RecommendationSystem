import os
import pandas as pd
import numpy as np
import tensorflow as tf
from recommender_system import KeekRecommendationSystem

def main():
    print("🚀 STARTING MODEL VERIFICATION")
    print("="*60)
    
    # Initialize system
    recommender = KeekRecommendationSystem()
    
    # Load artifacts
    print("\n📥 Loading model and artifacts...")
    try:
        recommender.load_artifacts(
            model_path='two_tower_model_base.h5',
            artifacts_path='model_artifacts_base.pkl'
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load data for testing
    print("\n📊 Loading production data for testing...")
    try:
        interaction_df, user_df, post_df = recommender.load_production_data()
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
        
    # Preprocess (Inference Mode)
    print("\n⚙️ Preprocessing data...")
    user_processed = recommender.preprocess_user_data(user_df, is_training=False)
    # For posts, we need to handle the fact that we might not have 'likes' etc for new posts in a real scenario,
    # but here we are testing on existing posts, so we can use the helper or just reuse logic.
    # Let's use the helper but be careful about data leakage if we were strictly evaluating.
    # For a quick "does it work" test, this is fine.
    post_processed = recommender.preprocess_post_data(post_df, interaction_df, is_training=False)

    # Interactive Loop
    while True:
        print("\n" + "="*60)
        print("🧪 TEST PREDICTION")
        print("="*60)
        
        # Pick a random user
        random_user = user_processed.sample(1).iloc[0]
        user_id = random_user['user_id']
        print(f"👤 Selected User ID: {user_id}")
        print(f"   Country: {random_user['country']} (Encoded: {random_user['country_encoded']})")
        print(f"   Age (Norm): {random_user['age_normalized']:.2f}")
        
        # Pick 5 random posts
        sample_posts = post_processed.sample(5)
        print(f"\n🎬 Scoring 5 Random Posts...")
        
        # Prepare inputs
        # We need to repeat user features for each post
        user_feats = np.tile(
            [random_user['country_encoded'], random_user['lang_encoded'], random_user['age_normalized']], 
            (5, 1)
        )
        
        post_feats = sample_posts[['country_encoded', 'lang_encoded', 'post_owner_encoded']].values
        
        # Predict
        predictions = recommender.model.predict(
            [user_feats, post_feats],
            verbose=0
        )
        
        # Display results
        print("\n📈 Predictions (Engagement Score 0-1):")
        results = []
        for i, (idx, post) in enumerate(sample_posts.iterrows()):
            score = predictions[i][0]
            results.append((post['post_id'], score, post['likes']))
            
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        for post_id, score, likes in results:
            print(f"   Post {post_id}: Score {score:.4f} (Actual Likes: {likes})")
            
        choice = input("\n🔄 Test another user? (y/n): ").lower()
        if choice != 'y':
            break

    print("\n✅ Verification finished.")

if __name__ == "__main__":
    main()
