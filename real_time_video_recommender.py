# -*- coding: utf-8 -*-
"""real_time_video_recommender.py

Real-Time Video Recommendation System
Automatically processes all videos in new_videos folder
"""

import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FIXED IMPORTS
# ============================================================
try:
    # Import your main recommendation system
    from main import KeekRecommendationSystem
    print("✅ KeekRecommendationSystem imported successfully from main.py")
except ImportError as e:
    print(f"❌ Could not import from main.py: {e}")
    try:
        from main import KeekRecommendationSystem
        print("✅ KeekRecommendationSystem imported successfully from complete_recommendation_system.py")
    except ImportError:
        print("❌ Could not import KeekRecommendationSystem from any file")
        exit(1)

class RealTimeVideoRecommender:
    def __init__(self, main_recommender=None):
        """
        Initialize Real-Time Video Recommender - FIXED VERSION
        
        Args:
            main_recommender: Pre-initialized KeekRecommendationSystem instance
        """
        if main_recommender is None:
            print("🔄 Initializing main recommendation system...")
            self.main_recommender = KeekRecommendationSystem()
            
            # FIX: Load existing data without reprocessing all training videos
            interaction_csv = "data/interaction_table_with_video_data.csv"
            if os.path.exists(interaction_csv):
                self.main_recommender.interaction_df = pd.read_csv(interaction_csv)
                print("✅ Loaded existing interaction data with video tags")
                
                # Build user profiles from existing data (no video processing)
                self.main_recommender.build_user_profiles(self.main_recommender.interaction_df)
                print(f"✅ Built user profiles for {len(self.main_recommender.user_profiles)} users")
            else:
                # Fallback: Full processing if no existing data
                print("⚠️ No existing data found, performing full processing...")
                self.main_recommender.get_user_profiles()
        else:
            self.main_recommender = main_recommender
        
        print("✅ Real-Time Video Recommender initialized!")
        print(f"   Loaded user profiles: {len(self.main_recommender.user_profiles)} users")
    
    def recommend_for_video(self, video_path, video_processor):
        """
        Complete pipeline for single video recommendation
        
        Args:
            video_path: Path to the new video file
            video_processor: Initialized VideoProcessingService instance
        
        Returns:
            Dictionary with video analysis and recommendations
        """
        print("🚀 PROCESSING VIDEO FOR REAL-TIME RECOMMENDATION")
        print("=" * 60)
        
        # Validate video file
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None
        
        print(f"📹 Video: {os.path.basename(video_path)}")
        print(f"📁 Path: {video_path}")
        
        try:
            # Step 1: Analyze video content using existing video processor
            print("\n🎬 Analyzing video content...")
            video_analysis = video_processor.analyze_single_video(video_path)
            
            if not video_analysis:
                print("❌ Video analysis failed")
                return None
            
            # Step 2: Get recommendations using main system's user profiles
            print("🔍 Finding matching users...")
            recommendations = self.main_recommender.recommend_for_new_video(video_analysis)
            
            # Step 3: Prepare results
            results = {
                'video_analysis': video_analysis,
                'recommendations': recommendations,
                'top_matches': recommendations[:10],
                'processed_at': datetime.now().isoformat(),
                'video_path': video_path,
                'video_filename': os.path.basename(video_path)
            }
            
            # Step 4: Display results
            self._display_recommendation_results(results)
            
            # Step 5: Save to real-time recommendations log
            self._save_to_realtime_log(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_all_videos(self, video_processor=None):
        """
        Automatically find and process all videos in new_videos folder
        
        Args:
            video_processor: Initialized VideoProcessingService instance
        
        Returns:
            Dictionary with all video recommendations
        """
        videos_folder = "new_videos"
        print(f"📁 SCANNING FOR VIDEOS IN: {videos_folder}")
        print("=" * 60)
        
        # Check if videos folder exists
        if not os.path.exists(videos_folder):
            print(f"❌ Videos folder not found: {videos_folder}")
            print("💡 Creating new_videos folder...")
            os.makedirs(videos_folder, exist_ok=True)
            print(f"✅ Created folder: {videos_folder}")
            print("📝 Please add video files to this folder and run again")
            return {}
        
        # Find all video files
        video_files = self._find_video_files(videos_folder)
        
        if not video_files:
            print("❌ No video files found in folder")
            print("💡 Supported formats: .mp4, .avi, .mov, .mkv")
            return {}
        
        print(f"🎬 Found {len(video_files)} video files:")
        for i, video_file in enumerate(video_files, 1):
            print(f"   {i}. {os.path.basename(video_file)}")
        
        # Process all videos
        all_results = {}
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'='*50}")
            print(f"📹 PROCESSING VIDEO {i}/{len(video_files)}")
            print(f"{'='*50}")
            
            results = self.recommend_for_video(video_path, video_processor)
            
            if results:
                video_id = f"video_{i}_{os.path.basename(video_path)}"
                all_results[video_id] = results
        
        # Export batch results
        if all_results:
            self.export_batch_recommendations(all_results)
            print(f"\n✅ Successfully processed {len(all_results)} videos!")
        else:
            print(f"\n❌ No videos were processed successfully")
        
        return all_results
    
    def _find_video_files(self, folder_path):
        """
        Find all video files in a folder
        
        Args:
            folder_path: Path to folder to search
        
        Returns:
            List of video file paths
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
        video_files = []
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Check if file has video extension
                if any(filename.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(file_path)
        
        # Sort for consistent processing order
        video_files.sort()
        return video_files
    
    def _display_recommendation_results(self, results):
        """Display detailed recommendation results"""
        print(f"\n🎉 REAL-TIME RECOMMENDATION RESULTS")
        print("=" * 50)
        
        video_analysis = results['video_analysis']
        recommendations = results['recommendations']
        
        print(f"📹 Video: {results['video_filename']}")
        print(f"🏷️ Content Tags: {video_analysis['predicted_tag_1']}, {video_analysis['predicted_tag_2']}")
        print(f"⏱️ Duration: {video_analysis['duration_sec']}s")
        print(f"👥 Matched Users: {len(recommendations)}")
        print(f"🕒 Processed: {results['processed_at']}")
        
        if recommendations:
            print(f"\n🏆 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(results['top_matches'][:10], 1):
                print(f"   {i:2d}. User {rec['user_id']:6} | Score: {rec['match_score']:.3f} | Level: {rec['recommendation_level']:15}")
                print(f"       Matching: {', '.join(rec['matching_tags'])}")
                print(f"       Prefers: {', '.join([f'{tag}({score:.2f})' for tag, score in rec['user_top_preferences'][:2]])}")
                
                if i == 5:  # Show only top 5 in detail
                    remaining = len(recommendations) - 5
                    if remaining > 0:
                        print(f"       ... and {remaining} more users")
                    break
        else:
            print("❌ No suitable user matches found for this video content")
        
        # Show recommendation distribution
        if recommendations:
            levels = {}
            for rec in recommendations:
                level = rec['recommendation_level']
                levels[level] = levels.get(level, 0) + 1
            
            print(f"\n📊 Recommendation Distribution:")
            for level, count in levels.items():
                print(f"   {level}: {count} users")
    
    def _save_to_realtime_log(self, results, filename='real_time_recommendations_log.csv'):
        """Save individual recommendation to real-time log"""
        log_data = []
        
        video_analysis = results['video_analysis']
        
        for rec in results['recommendations']:
            log_data.append({
                'timestamp': results['processed_at'],
                'video_filename': results['video_filename'],
                'video_path': results['video_path'],
                'video_tags': f"{video_analysis['predicted_tag_1']},{video_analysis['predicted_tag_2']}",
                'video_duration': video_analysis.get('duration_sec', 0),
                'user_id': rec['user_id'],
                'match_score': rec['match_score'],
                'matching_tags': ','.join(rec['matching_tags']),
                'user_preferences': ','.join([f"{tag}({score:.2f})" for tag, score in rec['user_top_preferences']]),
                'recommendation_level': rec['recommendation_level']
            })
        
        # Append to CSV log
        log_df = pd.DataFrame(log_data)
        file_exists = os.path.exists(filename)
        
        if file_exists:
            log_df.to_csv(filename, mode='a', header=False, index=False)
        else:
            log_df.to_csv(filename, index=False)
        
        print(f"💾 Recommendation logged to: {filename}")
    
    def export_batch_recommendations(self, batch_results, filename='batch_video_recommendations.csv'):
        """Export batch processing results to CSV"""
        print(f"\n💾 Exporting batch recommendations to {filename}...")
        
        all_recommendations = []
        
        for video_id, data in batch_results.items():
            video_analysis = data['video_analysis']
            
            for rec in data['recommendations']:
                all_recommendations.append({
                    'video_id': video_id,
                    'video_filename': data['video_filename'],
                    'video_tags': f"{video_analysis['predicted_tag_1']},{video_analysis['predicted_tag_2']}",
                    'video_duration': video_analysis.get('duration_sec', 0),
                    'user_id': rec['user_id'],
                    'match_score': rec['match_score'],
                    'matching_tags': ','.join(rec['matching_tags']),
                    'user_preferences': ','.join([f"{tag}({score:.2f})" for tag, score in rec['user_top_preferences']]),
                    'recommendation_level': rec['recommendation_level'],
                    'processed_at': data['processed_at']
                })
        
        recommendations_df = pd.DataFrame(all_recommendations)
        recommendations_df.to_csv(filename, index=False)
        
        print(f"✅ Batch recommendations saved to {filename}")
        print(f"   Total recommendations: {len(recommendations_df)}")
        print(f"   Unique videos: {recommendations_df['video_id'].nunique()}")
        print(f"   Unique users: {recommendations_df['user_id'].nunique()}")
        
        return recommendations_df


# ============================================================
# SIMPLE USAGE FUNCTIONS
# ============================================================

def process_all_videos():
    """
    One-line function to process all videos in new_videos folder
    
    Returns:
        Recommendation results for all videos
    """
    recommender = RealTimeVideoRecommender()
    
    try:
        from video_processing_service import VideoProcessingService
        video_processor = VideoProcessingService()
    except ImportError:
        print("❌ Could not import VideoProcessingService")
        return None
    
    return recommender.process_all_videos(video_processor)


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    # Initialize recommender
    real_time_recommender = RealTimeVideoRecommender()
    
    try:
        from video_processing_service import VideoProcessingService
        video_processor = VideoProcessingService()
        print("✅ VideoProcessingService imported successfully")
    except ImportError as e:
        print(f"❌ Could not import VideoProcessingService: {e}")
        print("💡 Make sure video_processing_service.py is in the same directory")
        exit(1)
    
    # Automatically process all videos in new_videos folder
    results = real_time_recommender.process_all_videos(video_processor)
    
    if results:
        print(f"\n🎉 SUCCESS: Processed {len(results)} videos automatically!")
    else:
        print(f"\n❌ No videos were processed")