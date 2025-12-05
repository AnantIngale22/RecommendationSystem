import redis
import json
import os

class RedisService:
    def __init__(self, host='localhost', port=6379, db=0):
        self.enabled = True
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            # Test connection
            self.client.ping()
            print("✅ Connected to Redis successfully.")
        except redis.ConnectionError:
            print("⚠️ Could not connect to Redis. Caching will be disabled.")
            self.enabled = False
            
    def cache_recommendations(self, user_id, recommendations, ttl_seconds=86400):
        """
        Cache recommendations for a user.
        recommendations: List of dicts
        ttl_seconds: Time to live (default 24 hours)
        """
        if not self.enabled:
            return
            
        key = f"rec:user:{user_id}"
        try:
            # Store as JSON string
            data = json.dumps(recommendations)
            self.client.setex(key, ttl_seconds, data)
            # print(f"   Saved to Redis: {key}")
        except Exception as e:
            print(f"❌ Failed to cache for user {user_id}: {e}")
            
    def get_recommendations(self, user_id):
        """Retrieve cached recommendations for a user"""
        if not self.enabled:
            return None
            
        key = f"rec:user:{user_id}"
        try:
            data = self.client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            print(f"❌ Failed to retrieve for user {user_id}: {e}")
            
        return None

    def update_video_score(self, video_id, score_delta):
        """
        Increment the popularity score of a video in the 'trending_videos' Sorted Set.
        video_id: The ID of the video
        score_delta: How much to add (e.g., +1 for Like, +0.1 for View)
        """
        if not self.enabled:
            return
            
        key = "trending_videos"
        try:
            # ZINCRBY: Increment the score of member in the sorted set stored at key
            self.client.zincrby(key, score_delta, video_id)
        except Exception as e:
            print(f"❌ Failed to update score for video {video_id}: {e}")

    def get_trending_videos(self, limit=10):
        """
        Get the top N trending videos from the Sorted Set.
        Returns a list of (video_id, score) tuples.
        """
        if not self.enabled:
            return []
            
        key = "trending_videos"
        try:
            # ZREVRANGE: Return range of members in sorted set, by index, with scores ordered from high to low
            return self.client.zrevrange(key, 0, limit - 1, withscores=True)
        except Exception as e:
            print(f"❌ Failed to get trending videos: {e}")
            return []
