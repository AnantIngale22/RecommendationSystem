# -*- coding: utf-8 -*-
"""train_base_model.py

Offline Training Script
Trains the Two-Tower model on historical production data and saves artifacts.
"""

import os
import argparse
from recommender_system import KeekRecommendationSystem

def main():
    print("🚀 STARTING OFFLINE TRAINING JOB")
    print("="*60)
    
    # Initialize system
    recommender = KeekRecommendationSystem()
    
    try:
        # Step 1: Load data
        # We use use_fresh_data=False to load from local CSVs in production_data/
        print("\n📥 STEP 1: Loading historical data...")
        
        # Check for sample flag
        parser = argparse.ArgumentParser()
        parser.add_argument('--sample', type=int, help='Sample size for quick testing')
        args = parser.parse_args()
        
        interaction_df, user_df, post_df = recommender.load_and_integrate_data(use_fresh_data=False)
        
        if args.sample:
            print(f"⚠️ SAMPLING MODE: Using {args.sample} rows for testing")
            interaction_df = interaction_df.sample(n=min(args.sample, len(interaction_df)), random_state=42)
        
        # Step 2: Preprocess data (Training Mode)
        print("\n⚙️ STEP 2: Preprocessing data...")
        user_processed = recommender.preprocess_user_data(user_df, is_training=True)
        post_processed = recommender.preprocess_post_data(post_df, interaction_df, is_training=True)
        
        # Step 3: Create training data
        print("\n📊 STEP 3: Creating training dataset...")
        training_data = recommender.create_training_data(interaction_df, user_processed, post_processed)
        
        # Step 4: Build and Train Model
        print("\n🧠 STEP 4: Building and training model...")
        recommender.build_two_tower_model(user_processed, post_processed)
        
        train_inputs, test_inputs = recommender.prepare_model_inputs(training_data)
        
        # Train for a few epochs
        history = recommender.train_model(train_inputs, test_inputs, epochs=5)
        
        # Step 5: Save Artifacts
        print("\n💾 STEP 5: Saving model and artifacts...")
        recommender.save_artifacts(
            model_path='two_tower_model_base.h5',
            artifacts_path='model_artifacts_base.pkl'
        )
        
        print("\n🎉 OFFLINE TRAINING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error in training job: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
