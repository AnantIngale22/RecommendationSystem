from redis_service import RedisService

def check_trending():
    print("📡 Connecting to Redis...")
    redis_service = RedisService()
    
    if not redis_service.enabled:
        print("❌ Redis is disabled. Cannot check trending.")
        return

    print("\n🏆 TOP 10 TRENDING VIDEOS (Real-Time Scoreboard)")
    print("=" * 50)
    print(f"{'Rank':<5} | {'Video ID':<15} | {'Score':<10}")
    print("-" * 50)
    
    # Get top 10 from Redis
    trending = redis_service.get_trending_videos(limit=10)
    
    if not trending:
        print("⚠️ No trending data found (List is empty).")
    else:
        for rank, (video_id, score) in enumerate(trending, 1):
            print(f"#{rank:<4} | {video_id:<15} | {score:<10.1f}")
            
    print("=" * 50)

if __name__ == "__main__":
    check_trending()
