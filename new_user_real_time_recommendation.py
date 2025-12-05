# -*- coding: utf-8 -*-
"""new_user_real_time_recommendation_complete.py

Real-Time New User Registration and Recommendation System
COMPLETE FIXED VERSION - All methods included, no video processing
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import from the main recommendation system
try:
    from main import KeekRecommendationSystem
    print("✅ KeekRecommendationSystem imported successfully")
except ImportError as e:
    print(f"❌ Could not import KeekRecommendationSystem: {e}")
    exit(1)

class RealTimeNewUserSystem:
    """
    Real-time system for handling new user registration and immediate recommendations
    COMPLETE VERSION - All methods included
    """
    
    def __init__(self, data_refresh_interval=24):  # hours
        self.recommender = KeekRecommendationSystem()
        self.popular_posts = None
        self.demographic_profiles = None
        self.last_data_refresh = None
        self.data_refresh_interval = data_refresh_interval
        self.new_users_log = 'new_users_registration_log.csv'
        self.real_time_recommendations_log = 'real_time_new_user_recommendations.csv'
        
        # Initialize system WITHOUT video processing
        self._initialize_system_fast()
    
    def _initialize_system_fast(self):
        """Initialize the recommendation system WITHOUT video processing"""
        print("🚀 INITIALIZING REAL-TIME NEW USER SYSTEM (FAST MODE)")
        print("=" * 60)
        print("📝 SKIPPING VIDEO PROCESSING - Using pre-existing data only")
        print("=" * 60)
        
        try:
            # Load data directly without video integration
            interaction_df, user_processed, post_processed = self._load_data_directly()
            
            # Build recommendation models
            self.build_popularity_model(interaction_df, post_processed)
            self.build_demographic_profiles(user_processed, interaction_df, post_processed)
            
            self.last_data_refresh = datetime.now()
            print("✅ Real-time system initialized successfully (FAST MODE)!")
            print(f"   Popular posts loaded: {len(self.popular_posts)}")
            print(f"   Demographic segments: {len(self.demographic_profiles)}")
            print(f"   Total users in system: {len(user_processed)}")
            print(f"   Total posts in system: {len(post_processed)}")
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            raise
    
    def _load_data_directly(self):
        """Load data directly from CSV files without video processing"""
        print("📊 Loading data directly from CSV files...")
        
        data_dir = "data"
        
        try:
            # Load interaction data
            interaction_path = os.path.join(data_dir, 'interaction_table_poc_output.csv')
            if not os.path.exists(interaction_path):
                # Try the video-enhanced version
                interaction_path = os.path.join(data_dir, 'interaction_table_with_video_data.csv')
                if not os.path.exists(interaction_path):
                    raise FileNotFoundError("No interaction data found")
            
            interaction_df = pd.read_csv(interaction_path)
            print(f"✅ Loaded interaction data: {interaction_df.shape}")
            
            # Load user data
            user_path = os.path.join(data_dir, 'user_table_poc_output.csv')
            user_df = pd.read_csv(user_path)
            print(f"✅ Loaded user data: {user_df.shape}")
            
            # Load post data
            post_path = os.path.join(data_dir, 'post_table_poc_output.csv')
            post_df = pd.read_csv(post_path)
            print(f"✅ Loaded post data: {post_df.shape}")
            
            # Preprocess data using existing methods
            user_processed = self.recommender.preprocess_user_data(user_df)
            post_processed = self.recommender.preprocess_post_data(post_df, interaction_df)
            
            # Build user profiles from existing interactions
            if hasattr(self.recommender, 'build_user_profiles'):
                self.recommender.build_user_profiles(interaction_df)
                print(f"✅ Built user profiles for {len(self.recommender.user_profiles)} users")
            else:
                # Create simple user profiles if method doesn't exist
                self._create_simple_user_profiles(interaction_df)
            
            return interaction_df, user_processed, post_df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Create sample data for demonstration
            return self._create_sample_data()
    
    def _create_simple_user_profiles(self, interaction_df):
        """Create simple user profiles if the main method is not available"""
        print("🔧 Creating simple user profiles...")
        
        self.recommender.user_profiles = {}
        
        # Simple profile creation based on interaction patterns
        for user_id in interaction_df['user_id'].unique():
            user_interactions = interaction_df[interaction_df['user_id'] == user_id]
            
            # Create basic profile with equal weights
            profile = {category: 1.0/len(self.recommender.TAGS) for category in self.recommender.TAGS}
            
            # Simple engagement-based adjustment
            total_engagement = len(user_interactions)
            if total_engagement > 0:
                # Slight preference for categories user interacted with
                for _, interaction in user_interactions.iterrows():
                    # Use available tags or default
                    tags = ['entertainment', 'vlog']  # Default tags
                    for tag in tags:
                        if tag in profile:
                            profile[tag] += 0.1
                
                # Normalize
                total = sum(profile.values())
                profile = {k: v/total for k, v in profile.items()}
            
            self.recommender.user_profiles[user_id] = profile
        
        print(f"✅ Created simple profiles for {len(self.recommender.user_profiles)} users")
    
    def _create_sample_data(self):
        """Create sample data if no data files exist"""
        print("⚠️ Creating sample data for demonstration...")
        
        # Sample interaction data
        interaction_data = []
        for i in range(100):
            interaction_data.append({
                'user_id': f"user_{np.random.randint(1, 50)}",
                'post_id': f"post_{np.random.randint(1, 40)}",
                'likes': np.random.randint(0, 10),
                'saves': np.random.randint(0, 5),
                'views': np.random.randint(0, 100)
            })
        
        interaction_df = pd.DataFrame(interaction_data)
        
        # Sample user data
        user_data = []
        countries = ['US', 'UK', 'CA', 'FR', 'DE', 'JP', 'IN', 'BR']
        languages = ['en', 'fr', 'de', 'es', 'hi', 'ja', 'pt']
        
        for i in range(50):
            user_data.append({
                'user_id': f"user_{i+1}",
                'country': np.random.choice(countries),
                'supported_language': f'["{np.random.choice(languages)}"]',
                'age': np.random.randint(18, 65)
            })
        
        user_df = pd.DataFrame(user_data)
        user_processed = self.recommender.preprocess_user_data(user_df)
        
        # Sample post data
        post_data = []
        for i in range(40):
            post_data.append({
                'post_id': f"post_{i+1}",
                'country': np.random.choice(countries),
                'lang': np.random.choice(languages),
                'post_owner_id': f"user_{np.random.randint(1, 50)}"
            })
        
        post_df = pd.DataFrame(post_data)
        
        # Create simple user profiles
        self._create_simple_user_profiles(interaction_df)
        
        print("✅ Sample data created for demonstration")
        return interaction_df, user_processed, post_df

    def build_popularity_model(self, interaction_df, post_processed, top_n=100):
        """Build popularity-based recommendation model (Hybrid: Redis + CSV)"""
        print("🏆 Building popularity-based recommendation model...")
        
        # 1. Try to get Real-Time Trending from Redis
        try:
            from redis_service import RedisService
            redis_service = RedisService()
            if redis_service.enabled:
                print("   📡 Checking Redis for real-time trending videos...")
                real_time_trending = redis_service.get_trending_videos(limit=top_n)
                
                if real_time_trending:
                    print(f"   ✅ Found {len(real_time_trending)} trending videos in Redis!")
                    
                    # Convert Redis format [(vid, score), ...] to DataFrame format
                    redis_scores = []
                    for vid, score in real_time_trending:
                        redis_scores.append({
                            'post_id': vid,
                            'engagement_score': score,
                            'normalized_score': score / 100.0, # Normalize arbitrarily for now
                            'total_interactions': int(score),
                            'avg_likes': 0, # We don't have this detail in simple Redis ZSET
                            'avg_saves': 0
                        })
                    
                    self.popular_posts = pd.DataFrame(redis_scores)
                    return self.popular_posts
                else:
                    print("   ⚠️ Redis trending list is empty. Falling back to CSV.")
        except Exception as e:
            print(f"   ⚠️ Redis check failed: {e}. Falling back to CSV.")

        # 2. Fallback to CSV (Static Calculation)
        print("   📊 Calculating trending from static CSV data...")
        
        # Calculate popularity scores for posts
        popularity_scores = []
        
        for post_id in post_processed['post_id'].unique():
            post_interactions = interaction_df[interaction_df['post_id'] == post_id]
            
            if len(post_interactions) > 0:
                # Calculate weighted engagement score
                engagement_score = (
                    post_interactions['likes'].sum() * self.recommender.WEIGHTS['likes'] +
                    post_interactions['saves'].sum() * self.recommender.WEIGHTS['saves'] +
                    post_interactions['views'].sum() * self.recommender.WEIGHTS['views']
                )
                
                # Normalize by number of interactions
                normalized_score = engagement_score / len(post_interactions)
                
                popularity_scores.append({
                    'post_id': post_id,
                    'engagement_score': engagement_score,
                    'normalized_score': normalized_score,
                    'total_interactions': len(post_interactions),
                    'avg_likes': post_interactions['likes'].mean(),
                    'avg_saves': post_interactions['saves'].mean()
                })
            else:
                # For posts with no interactions, use default score
                popularity_scores.append({
                    'post_id': post_id,
                    'engagement_score': 0,
                    'normalized_score': 0,
                    'total_interactions': 0,
                    'avg_likes': 0,
                    'avg_saves': 0
                })
        
        # Create popularity dataframe and sort
        popularity_df = pd.DataFrame(popularity_scores)
        popularity_df = popularity_df.sort_values('normalized_score', ascending=False)
        
        # Get top N popular posts
        self.popular_posts = popularity_df.head(top_n)
        
        print(f"✅ Popularity model built with {len(self.popular_posts)} popular posts (Source: CSV)")
        return self.popular_posts

    def build_demographic_profiles(self, user_processed, interaction_df, post_processed):
        """Build demographic-based recommendation profiles"""
        print("👥 Building demographic-based recommendation profiles...")
        
        demographic_profiles = {}
        
        # Group users by demographic segments
        for _, user in user_processed.iterrows():
            segment_key = f"{user['country_encoded']}_{user['lang_encoded']}"
            
            if segment_key not in demographic_profiles:
                demographic_profiles[segment_key] = {
                    'country_code': user['country_encoded'],
                    'language_code': user['lang_encoded'],
                    'user_ids': [],
                    'preferred_categories': defaultdict(float),
                    'popular_posts': defaultdict(float),
                    'segment_size': 0
                }
            
            demographic_profiles[segment_key]['user_ids'].append(user['user_id'])
            demographic_profiles[segment_key]['segment_size'] += 1
        
        # Calculate preferences for each demographic segment
        for segment_key, segment_data in demographic_profiles.items():
            segment_users = segment_data['user_ids']
            
            # Get interactions for users in this segment
            segment_interactions = interaction_df[interaction_df['user_id'].isin(segment_users)]
            
            if len(segment_interactions) > 0:
                # Use default tags since we're skipping video processing
                default_tags = ['entertainment', 'vlog']
                
                for _, interaction in segment_interactions.iterrows():
                    engagement_score = (
                        interaction['likes'] * self.recommender.WEIGHTS['likes'] +
                        interaction['saves'] * self.recommender.WEIGHTS['saves'] +
                        interaction['views'] * self.recommender.WEIGHTS['views']
                    )
                    
                    for tag in default_tags:
                        if tag in self.recommender.TAGS:
                            segment_data['preferred_categories'][tag] += engagement_score
                
                # Calculate popular posts within segment
                post_engagement = segment_interactions.groupby('post_id').agg({
                    'likes': 'sum',
                    'saves': 'sum', 
                    'views': 'sum'
                }).reset_index()
                
                for _, post_row in post_engagement.iterrows():
                    post_score = (
                        post_row['likes'] * self.recommender.WEIGHTS['likes'] +
                        post_row['saves'] * self.recommender.WEIGHTS['saves'] +
                        post_row['views'] * self.recommender.WEIGHTS['views']
                    )
                    segment_data['popular_posts'][post_row['post_id']] = post_score
            
            # Normalize category preferences
            total_category_score = sum(segment_data['preferred_categories'].values())
            if total_category_score > 0:
                for category in segment_data['preferred_categories']:
                    segment_data['preferred_categories'][category] /= total_category_score
        
        self.demographic_profiles = demographic_profiles
        print(f"✅ Demographic profiles built for {len(demographic_profiles)} segments")
        return demographic_profiles

    def register_new_user(self, user_data):
        """
        Register a new user and provide immediate recommendations
        """
        print(f"👤 REGISTERING NEW USER - Real-time processing...")
        print("=" * 50)
        
        # Validate required fields
        if not user_data.get('country') or not user_data.get('language'):
            return {
                'success': False,
                'error': 'Country and language are required fields',
                'user_id': None,
                'recommendations': []
            }
        
        try:
            # Generate unique user ID
            user_id = self._generate_user_id()
            
            # Add metadata
            user_data['user_id'] = user_id
            user_data['registration_timestamp'] = datetime.now().isoformat()
            user_data['ip_address'] = user_data.get('ip_address', 'unknown')
            user_data['signup_source'] = user_data.get('signup_source', 'direct')
            
            # Log user registration
            self._log_user_registration(user_data)
            
            # Generate immediate recommendations
            recommendations = self.recommend_for_new_user(
                user_country=user_data['country'],
                user_language=user_data['language'],
                user_age=user_data.get('age', 25),
                top_k=15,
                popularity_weight=0.3,
                demographic_weight=0.7
            )
            
            # Prepare response
            response = {
                'success': True,
                'user_id': user_id,
                'registration_timestamp': user_data['registration_timestamp'],
                'welcome_message': self._generate_welcome_message(user_data),
                'recommendations': recommendations,
                'recommendation_count': len(recommendations),
                'demographic_segment': self._get_demographic_segment_key(
                    user_data['country'], user_data['language']
                )
            }
            
            # Log recommendations
            self._log_real_time_recommendations(user_id, recommendations)
            
            print(f"✅ User {user_id} registered successfully!")
            print(f"   Recommendations generated: {len(recommendations)}")
            
            return response
            
        except Exception as e:
            print(f"❌ User registration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_id': None,
                'recommendations': []
            }

    def recommend_for_new_user(self, user_country="US", user_language="en", user_age=25, 
                              top_k=10, popularity_weight=0.4, demographic_weight=0.6):
        """
        Generate recommendations for new user combining popularity and demographic approaches
        """
        print(f"🎯 Generating real-time recommendations...")
        
        # Check if data needs refresh
        self._check_data_refresh()
        
        # Get demographic segment for the user
        demographic_segment = self.get_demographic_segment(
            user_country, user_language, self.recommender.user_processed
        )
        
        print(f"   Matched demographic segment: {demographic_segment}")
        
        # Get popularity-based recommendations
        popularity_recs = self._get_popularity_recommendations(top_k * 2)
        
        # Get demographic-based recommendations
        demographic_recs = self._get_demographic_recommendations(demographic_segment, top_k * 2)
        
        # Combine recommendations using weighted scores
        combined_recommendations = self._combine_recommendations(
            popularity_recs, 
            demographic_recs, 
            popularity_weight, 
            demographic_weight,
            top_k
        )
        
        # Add explanation for recommendations
        recommendations_with_explanation = self._add_recommendation_explanations(
            combined_recommendations, demographic_segment
        )
        
        print(f"✅ Generated {len(recommendations_with_explanation)} recommendations")
        return recommendations_with_explanation

    def get_demographic_segment(self, user_country, user_language, user_processed):
        """Find the best demographic segment for a new user"""
        # Encode country and language
        country_encoded = self._encode_country(user_country, user_processed)
        language_encoded = self._encode_language(user_language, user_processed)
        
        segment_key = f"{country_encoded}_{language_encoded}"
        
        # Try exact match first
        if segment_key in self.demographic_profiles:
            return segment_key
        
        # Try country match with different language
        for key in self.demographic_profiles.keys():
            if key.startswith(f"{country_encoded}_"):
                return key
        
        # Try language match with different country  
        for key in self.demographic_profiles.keys():
            if key.endswith(f"_{language_encoded}"):
                return key
        
        # Return most common segment as fallback
        return max(self.demographic_profiles.keys(), 
                  key=lambda k: self.demographic_profiles[k]['segment_size'])

    def _encode_country(self, country, user_processed):
        """Encode country to match existing encoding"""
        if 'country' in user_processed.columns:
            unique_countries = user_processed['country'].unique()
            if country in unique_countries:
                return list(unique_countries).index(country)
        return 0  # Default encoding

    def _encode_language(self, language, user_processed):
        """Encode language to match existing encoding"""
        if 'primary_language' in user_processed.columns:
            unique_languages = user_processed['primary_language'].unique()
            if language in unique_languages:
                return list(unique_languages).index(language)
        return 0  # Default encoding

    def _get_popularity_recommendations(self, top_k):
        """Get popularity-based recommendations"""
        recommendations = []
        
        for _, post in self.popular_posts.head(top_k).iterrows():
            recommendations.append({
                'post_id': post['post_id'],
                'score': post['normalized_score'],
                'type': 'popularity',
                'reason': 'Trending content with high engagement',
                'engagement_metrics': {
                    'avg_likes': post['avg_likes'],
                    'avg_saves': post['avg_saves'],
                    'total_interactions': post['total_interactions']
                }
            })
        
        return recommendations

    def _get_demographic_recommendations(self, demographic_segment, top_k):
        """Get demographic-based recommendations"""
        recommendations = []
        
        if demographic_segment in self.demographic_profiles:
            segment_data = self.demographic_profiles[demographic_segment]
            
            # Get top categories for this segment
            top_categories = sorted(
                segment_data['preferred_categories'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            # Get popular posts within segment
            popular_posts = sorted(
                segment_data['popular_posts'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            for post_id, score in popular_posts:
                recommendations.append({
                    'post_id': post_id,
                    'score': score,
                    'type': 'demographic',
                    'reason': f'Popular among users from your demographic segment',
                    'top_categories': top_categories,
                    'segment_size': segment_data['segment_size']
                })
        
        return recommendations

    def _combine_recommendations(self, popularity_recs, demographic_recs, 
                               pop_weight, demo_weight, top_k):
        """Combine popularity and demographic recommendations"""
        combined_scores = {}
        
        # Add popularity recommendations with weight
        for rec in popularity_recs:
            post_id = rec['post_id']
            combined_scores[post_id] = {
                'popularity_score': rec['score'] * pop_weight,
                'demographic_score': 0,
                'popularity_data': rec,
                'demographic_data': None
            }
        
        # Add demographic recommendations with weight
        for rec in demographic_recs:
            post_id = rec['post_id']
            if post_id in combined_scores:
                combined_scores[post_id]['demographic_score'] = rec['score'] * demo_weight
                combined_scores[post_id]['demographic_data'] = rec
            else:
                combined_scores[post_id] = {
                    'popularity_score': 0,
                    'demographic_score': rec['score'] * demo_weight,
                    'popularity_data': None,
                    'demographic_data': rec
                }
        
        # Calculate combined scores and prepare final recommendations
        final_recommendations = []
        for post_id, scores in combined_scores.items():
            total_score = scores['popularity_score'] + scores['demographic_score']
            
            recommendation = {
                'post_id': post_id,
                'combined_score': total_score,
                'popularity_contribution': scores['popularity_score'],
                'demographic_contribution': scores['demographic_score']
            }
            
            # Add reason based on which approach contributed more
            if scores['popularity_score'] > scores['demographic_score']:
                recommendation['primary_reason'] = 'popularity'
                if scores['popularity_data']:
                    recommendation.update(scores['popularity_data'])
            else:
                recommendation['primary_reason'] = 'demographic'
                if scores['demographic_data']:
                    recommendation.update(scores['demographic_data'])
            
            final_recommendations.append(recommendation)
        
        # Sort by combined score and return top K
        final_recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        return final_recommendations[:top_k]

    def _add_recommendation_explanations(self, recommendations, demographic_segment):
        """Add detailed explanations for recommendations"""
        for rec in recommendations:
            if rec['primary_reason'] == 'popularity':
                rec['explanation'] = (
                    f"This content is trending with high engagement across all users. "
                    f"It has {rec.get('engagement_metrics', {}).get('total_interactions', 0)} interactions."
                )
            else:
                segment_data = self.demographic_profiles.get(demographic_segment, {})
                top_categories = rec.get('top_categories', [])
                
                if top_categories:
                    category_names = [cat[0] for cat in top_categories[:2]]
                    rec['explanation'] = (
                        f"Users similar to you (demographic segment: {demographic_segment}) "
                        f"frequently engage with {', '.join(category_names)} content. "
                        f"This post is popular among {segment_data.get('segment_size', 0)} similar users."
                    )
                else:
                    rec['explanation'] = (
                        f"Recommended based on preferences of users in your demographic segment "
                        f"({demographic_segment})."
                    )
        
        return recommendations

    def _generate_user_id(self):
        """Generate unique user ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = np.random.randint(1000, 9999)
        return f"user_{timestamp}_{random_suffix}"

    def _generate_welcome_message(self, user_data):
        """Generate personalized welcome message"""
        country = user_data.get('country', '')
        language = user_data.get('language', '')
        
        messages = {
            'en': f"Welcome to our platform! We're excited to have you from {country}.",
            'fr': f"Bienvenue sur notre plateforme! Heureux de vous accueillir de {country}.",
            'es': f"¡Bienvenido a nuestra plataforma! Estamos encantados de tenerte desde {country}.",
            'de': f"Willkommen auf unserer Plattform! Wir freuen uns, Sie aus {country} zu haben."
        }
        
        return messages.get(language, messages['en'])

    def _get_demographic_segment_key(self, country, language):
        """Get demographic segment key for user"""
        country_encoded = self._encode_country(country, self.recommender.user_processed)
        language_encoded = self._encode_language(language, self.recommender.user_processed)
        return f"{country_encoded}_{language_encoded}"

    def _log_user_registration(self, user_data):
        """Log new user registration to CSV"""
        log_entry = {
            'user_id': user_data['user_id'],
            'registration_timestamp': user_data['registration_timestamp'],
            'country': user_data.get('country', ''),
            'language': user_data.get('language', ''),
            'age': user_data.get('age', ''),
            'ip_address': user_data.get('ip_address', ''),
            'signup_source': user_data.get('signup_source', ''),
            'demographic_segment': self._get_demographic_segment_key(
                user_data.get('country', ''), user_data.get('language', '')
            )
        }
        
        # Append to CSV
        log_df = pd.DataFrame([log_entry])
        file_exists = os.path.exists(self.new_users_log)
        
        if file_exists:
            log_df.to_csv(self.new_users_log, mode='a', header=False, index=False)
        else:
            log_df.to_csv(self.new_users_log, index=False)
        
        print(f"📝 User registration logged: {user_data['user_id']}")

    def _log_real_time_recommendations(self, user_id, recommendations):
        """Log real-time recommendations to CSV"""
        log_data = []
        timestamp = datetime.now().isoformat()
        
        for i, rec in enumerate(recommendations, 1):
            log_data.append({
                'timestamp': timestamp,
                'user_id': user_id,
                'post_id': rec['post_id'],
                'rank': i,
                'combined_score': rec['combined_score'],
                'popularity_contribution': rec['popularity_contribution'],
                'demographic_contribution': rec['demographic_contribution'],
                'primary_reason': rec['primary_reason'],
                'recommendation_type': rec.get('type', 'combined'),
                'explanation': rec.get('explanation', '')
            })
        
        # Append to CSV
        log_df = pd.DataFrame(log_data)
        file_exists = os.path.exists(self.real_time_recommendations_log)
        
        if file_exists:
            log_df.to_csv(self.real_time_recommendations_log, mode='a', header=False, index=False)
        else:
            log_df.to_csv(self.real_time_recommendations_log, index=False)
        
        print(f"📊 Recommendations logged for user: {user_id}")

    def _check_data_refresh(self):
        """Check if data needs to be refreshed"""
        if self.last_data_refresh is None:
            return
        
        hours_since_refresh = (datetime.now() - self.last_data_refresh).total_seconds() / 3600
        
        if hours_since_refresh >= self.data_refresh_interval:
            print("🔄 Refreshing recommendation data...")
            self._initialize_system_fast()

    def get_system_stats(self):
        """Get current system statistics"""
        return {
            'system_status': 'active',
            'last_refresh': self.last_data_refresh.isoformat() if self.last_data_refresh else 'never',
            'popular_posts_loaded': len(self.popular_posts) if self.popular_posts else 0,
            'demographic_segments': len(self.demographic_profiles) if self.demographic_profiles else 0,
            'data_refresh_interval_hours': self.data_refresh_interval
        }

def demonstrate_real_time_system_fast():
    """Demonstrate the FAST real-time new user recommendation system"""
    print("🎉 REAL-TIME NEW USER RECOMMENDATION SYSTEM (FAST MODE)")
    print("=" * 60)
    print("📝 NO VIDEO PROCESSING - Using existing data only")
    print("=" * 60)
    
    # Initialize system
    print("🚀 Initializing system in FAST mode...")
    system = RealTimeNewUserSystem()
    
    # Test cases
    test_users = [
        {
            'country': 'US',
            'language': 'en',
            'age': 28,
            'ip_address': '192.168.1.100',
            'signup_source': 'web'
        },
        {
            'country': 'FR', 
            'language': 'fr',
            'age': 32,
            'ip_address': '89.156.45.23',
            'signup_source': 'mobile_app'
        }
    ]
    
    for i, user_data in enumerate(test_users, 1):
        print(f"\n{'='*50}")
        print(f"👤 TEST USER {i}: {user_data['country']}-{user_data['language']}")
        print(f"{'='*50}")
        
        result = system.register_new_user(user_data)
        
        if result['success']:
            print(f"✅ SUCCESS: User {result['user_id']}")
            print(f"📊 Recommendations: {result['recommendation_count']}")
            print(f"🎯 Welcome: {result['welcome_message']}")
            
            # Show top 3 recommendations
            print(f"\n🏆 TOP 3 RECOMMENDATIONS:")
            for j, rec in enumerate(result['recommendations'][:3], 1):
                print(f"   {j}. Post {rec['post_id']} - Score: {rec['combined_score']:.3f}")
                print(f"      Type: {rec['primary_reason']}")
                print(f"      Explanation: {rec.get('explanation', '')[:80]}...")
        else:
            print(f"❌ FAILED: {result['error']}")
    
    print(f"\n🎊 FAST MODE DEMONSTRATION COMPLETED!")
    print("💾 Check generated files:")
    print("   - new_users_registration_log.csv")
    print("   - real_time_new_user_recommendations.csv")

if __name__ == "__main__":
    # Run the FAST version
    demonstrate_real_time_system_fast()