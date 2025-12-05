# test_real_time.py - Test real-time recommender
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import KeekRecommendationSystem
    print("✅ KeekRecommendationSystem imported successfully!")
    
    from video_processing_service import VideoProcessingService
    print("✅ VideoProcessingService imported successfully!")
    
    # Test the recommender
    recommender = KeekRecommendationSystem()
    recommender.get_user_profiles()  # This builds profiles once
    
    print("🎉 Real-time recommender is ready!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")