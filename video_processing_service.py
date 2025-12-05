import os
import json
import csv
import subprocess
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np

# ... (imports)

# ============================================================
# FUNCTIONS
# ============================================================
def trim_video_to_one_minute(video_path):
    """Trim the video to the first 1 minute (60s)."""
    # Use a temporary file in the same directory to avoid cross-device link errors
    temp_path = f"{video_path}_trimmed.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-t", "60", "-c", "copy", temp_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return temp_path

def get_video_duration(video_path):
    """Extract duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        return round(float(result.stdout.decode().strip()), 2)
    except:
        return 0.0

TAGS = [
    "sports", "gaming", "cooking", "traveling", "entertainment", "music",
    "education", "news", "vlog", "review", "fashion", "technology",
    "fitness", "food", "animals", "nature", "comedy", "lifestyle",
    "celebrity", "flowers"
]

NUM_FRAMES = 8
TOP_K = 3

# ============================================================
# MODEL LOADING
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

class VideoProcessingService:
    def __init__(self):
        self.TAGS = TAGS
    
    def get_video_files_from_folder(self, folder_path):
        """Get all video files from a folder location"""
        print(f"🔍 Scanning folder for video files: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return []

        # Get ALL files in directory
        all_files = os.listdir(folder_path)
        print(f"📄 Total files in folder: {len(all_files)}")
        
        # Filter for video files (case insensitive)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
        video_files = []
        
        for file in all_files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                # Check extension
                if any(file.lower().endswith(ext.lower()) for ext in video_extensions):
                    video_files.append(file_path)
                # Also check if file contains "video" in name (case insensitive)
                elif 'video' in file.lower() or 'keek' in file.lower():
                    video_files.append(file_path)
        
        # Sort files for consistent processing
        video_files.sort()
        
        print(f"🎯 Found {len(video_files)} video files")
        return video_files
    
    def process_video_batch(self, video_paths):
        """Process batch of videos and extract metadata"""
        print(f"🎥 Processing {len(video_paths)} videos...")
        
        results = []
        for i, video_path in enumerate(video_paths, 1):
            try:
                print(f"\n📊 Progress: {i}/{len(video_paths)}")
                video_data = self.analyze_single_video(video_path)
                results.append(video_data)
                
                # Clear memory
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"❌ Error processing {video_path}: {e}")
                continue
        
        print(f"✅ Processed {len(results)} videos successfully")
        return results
    
    def analyze_single_video(self, video_path):
        """Analyze single video and extract features using CLIP model"""
        print(f"🎞️ Processing: {os.path.basename(video_path)}")

        try:
            # Check if file exists and is readable
            if not os.path.exists(video_path):
                print(f"❌ File not found: {video_path}")
                return self._create_error_result(video_path)
            
            file_size = os.path.getsize(video_path)
            print(f"   📏 File size: {file_size / (1024*1024):.2f} MB")
            
            trimmed = trim_video_to_one_minute(video_path)
            duration = get_video_duration(trimmed)
            print(f"   ⏱️ Duration: {duration} seconds")

            # Extract frames using OpenCV
            cap = cv2.VideoCapture(trimmed)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"   🖼️ Total frames: {total_frames}")
            
            if total_frames <= 0:
                 print("⚠️ Could not read frames from video")
                 return self._create_error_result(video_path)

            step = max(1, total_frames // NUM_FRAMES)
            frame_indices = list(range(0, total_frames, step))[:NUM_FRAMES]
            print(f"   🔍 Analyzing {len(frame_indices)} frames")

            scores = torch.zeros(len(self.TAGS)).to(device)

            for idx in tqdm(frame_indices, desc="Analyzing frames", leave=False):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                inputs = processor(text=self.TAGS, images=frame, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)
                    scores += probs
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            cap.release()

            scores /= len(frame_indices)
            top_indices = scores.topk(TOP_K).indices.tolist()

            filename = os.path.basename(video_path)

            # ✅ Return format compatible with both systems
            result = {
                'video_path': video_path,
                'duration_sec': duration,
                'predicted_tag_1': self.TAGS[top_indices[0]],
                'predicted_tag_2': self.TAGS[top_indices[1]] if len(top_indices) > 1 else self.TAGS[top_indices[0]],
                'predicted_tags': [
                    {"tag": self.TAGS[i], "score": round(scores[i].item(), 3)} for i in top_indices
                ]
            }
            
            print(f"   🏷️ Predicted tags: {result['predicted_tag_1']}, {result['predicted_tag_2']}")
            return result
            
        except Exception as e:
            print(f"❌ Error in video analysis: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(video_path)
    
    def _create_error_result(self, video_path):
        """Create a basic result when video analysis fails"""
        return {
            'video_path': video_path,
            'duration_sec': 0,
            'predicted_tag_1': 'unknown',
            'predicted_tag_2': 'unknown',
            'predicted_tags': [{"tag": "unknown", "score": 0.0}]
        }
    
    def process_folder_videos(self, folder_path):
        """Process all videos in a folder"""
        video_files = self.get_video_files_from_folder(folder_path)
        return self.process_video_batch(video_files)

# ============================================================
# COMPATIBILITY WRAPPER FOR REAL-TIME RECOMMENDER
# ============================================================
class RealTimeCompatibleVideoProcessor:
    """Wrapper to make VideoProcessingService compatible with real-time recommender"""
    def __init__(self):
        self.video_service = VideoProcessingService()
    
    def analyze_single_video(self, video_path):
        """
        Analyze single video - compatible with real-time recommender
        Returns only the fields needed by the recommender
        """
        result = self.video_service.analyze_single_video(video_path)
        
        # Extract only the fields needed by real-time recommender
        return {
            'duration_sec': result['duration_sec'],
            'predicted_tag_1': result['predicted_tag_1'],
            'predicted_tag_2': result['predicted_tag_2']
        }

# ============================================================
# MAIN LOOP (Optional - for standalone video processing)
# ============================================================
def main():
    """Main function for standalone video processing"""
    service = VideoProcessingService()
    results = service.process_folder_videos("training_videos")
    print(f"Processed {len(results)} videos")

if __name__ == "__main__":
    """Main function to run video processing"""
    print("🎯 VIDEO CATEGORIES EXTRACTION SYSTEM")
    print("="*50)
    print(f"🎯 PROCESSING VIDEOS FROM LOCAL FOLDER")
    print("="*50)
    
    # Check if video directory exists
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ Video directory not found: {VIDEO_DIR}")
    
    
    # Initialize video processing service
    video_service = VideoProcessingService()
    
    # Process videos from folder using the service
    print(f"📂 Processing videos from: {VIDEO_DIR}")
    processed_videos = video_service.process_folder_videos(VIDEO_DIR)
    
    if processed_videos:
        dataset = []
        for video in processed_videos:
            dataset.append({
                "path": video['video_path'],
                "duration_sec": video['duration_sec'],
                "predicted_tags": video['predicted_tags']
            })
        
        # Save JSON locally
        with open(OUTPUT_JSON, "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"\n✅ All video analyses saved to {OUTPUT_JSON}")

        # Save CSV locally
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "duration_sec", "predicted_tags"])
            for d in dataset:
                tags_str = ", ".join([f"{t['tag']} ({t['score']})" for t in d["predicted_tags"]])
                writer.writerow([d["path"], d["duration_sec"], tags_str])
        print(f"📄 CSV version saved to {OUTPUT_CSV}")
            
    else:
        print("❌ No videos were processed successfully")

if __name__ == "__main__":
    main()