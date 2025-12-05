import requests
import random
import time
import concurrent.futures

# Configuration
API_URL = "http://localhost:8000/interact"
NUM_REQUESTS = 100  # How many interactions to simulate
CONCURRENT_USERS = 10 # How many parallel threads

# Mock Data Pools
USER_IDS = [f"user_{i}" for i in range(1, 51)] # 50 Users
VIDEO_IDS = [f"video_{i}" for i in range(1, 21)] # 20 Videos
ACTIONS = ['view', 'view', 'view', 'like', 'like', 'save'] # Weighted: Mostly views

def send_interaction(i):
    """Send a single random interaction"""
    user = random.choice(USER_IDS)
    video = random.choice(VIDEO_IDS)
    action = random.choice(ACTIONS)
    
    try:
        response = requests.post(API_URL, data={
            "user_id": user,
            "video_id": video,
            "action": action
        })
        return f"✅ {user} -> {action} -> {video} ({response.status_code})"
    except Exception as e:
        return f"❌ Failed: {e}"

def main():
    print(f"🚀 Starting Simulation: {NUM_REQUESTS} interactions...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        futures = [executor.submit(send_interaction, i) for i in range(NUM_REQUESTS)]
        
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
            
    duration = time.time() - start_time
    print(f"\n🎉 Simulation Complete in {duration:.2f} seconds!")
    print(f"⚡️ Rate: {NUM_REQUESTS / duration:.2f} req/sec")

if __name__ == "__main__":
    main()
