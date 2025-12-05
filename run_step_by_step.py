from code.main import KeekRecommendationSystem

print("🎯 STEP 1: Initializing system...")
recommender = KeekRecommendationSystem()

print("🎯 STEP 2: Loading data...")
interaction_df, user_df, post_df = recommender.load_production_data()

print("🎯 STEP 3: Preprocessing data...")
user_processed = recommender.preprocess_user_data(user_df)
post_processed = recommender.preprocess_post_data(post_df, interaction_df)

print("🎯 STEP 4: Building user profiles...")
recommender.build_user_profiles(interaction_df)

print("🎯 STEP 5: Creating training data...")
training_data = recommender.create_training_data(interaction_df, user_processed, post_processed)

print("🎯 STEP 6: Building model...")
recommender.build_two_tower_model(user_processed, post_processed)

print("🎯 STEP 7: Training model...")
train_inputs, test_inputs = recommender.prepare_model_inputs(training_data)
history = recommender.train_model(train_inputs, test_inputs, epochs=15)

print("🎯 STEP 8: Generating recommendations...")
recommendations = recommender.generate_recommendations(user_processed, post_processed, interaction_df)

print("🎯 STEP 9: Saving results...")
recommender.save_recommendations(recommendations)
recommender.save_model()

print("🎉 ALL STEPS COMPLETED!")