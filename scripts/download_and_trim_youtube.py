import os
from pathlib import Path
import yt_dlp

# ================= CONFIGURATION =================

# 1. Where to save the files
DATA_DIR = Path(r"C:\Users\USER\Documents\meeting-asr\data\youtube_clips")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2. !!! CRITICAL FIX: POINT THIS TO YOUR FFMPEG BIN FOLDER !!!
FFMPEG_LOCATION = r"C:\Users\USER\Downloads\ffmpeg-8.0.1-essentials_build\bin" 

# === FORCE PATH UPDATE (The "Nuclear" Fix) ===
if os.path.exists(FFMPEG_LOCATION):
    os.environ["PATH"] += os.pathsep + FFMPEG_LOCATION
    print(f"DEBUG: Added FFmpeg to PATH: {FFMPEG_LOCATION}")
else:
    print(f"WARNING: FFmpeg path not found at: {FFMPEG_LOCATION}")
# ==============================================

# 3. List of videos to process
# Format: (URL, Output_Filename, Start_Time_Seconds, End_Time_Seconds)
DOWNLOAD_LIST = [
    # Keluar Sekejap (Malay/English Code-Switching) - EP140
    #("https://www.youtube.com/watch?v=J5pNJSstp6s", "keluar_sekejap_sample", 600, 900),
    
    # JinnyboyTV (Manglish / Urban English) - "How To Be Famous"
    #("https://www.youtube.com/watch?v=unqAzKXB6OM", "jinnyboy_sample", 60, 360),

    # 8TV Mandarin News (Formal Mandarin) - News Broadcast
    ("https://www.youtube.com/watch?v=skuqA7H8ZdM", "mandarin_news_sample", 360, 660)
]

# ================= LOGIC =================

def download_and_trim(url, filename, start_sec, end_sec):
    print(f"\n[Processing] {filename}...")
    output_path = DATA_DIR / f"{filename}.wav"
    
    def download_range_func(info_dict, ydl):
        if start_sec is None: return None
        end = end_sec if end_sec else info_dict.get('duration')
        return [{'start_time': start_sec, 'end_time': end}]

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(DATA_DIR / f"{filename}"),
        'ffmpeg_location': FFMPEG_LOCATION,
        
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        
        'postprocessor_args': ['-ar', '16000', '-ac', '1'],
        
        # FIX: Explicitly check against None. 
        # Previously "if start_sec" evaluated 0 as False, breaking the 3rd video.
        'download_ranges': download_range_func if start_sec is not None else None,
        
        'overwrites': True,
        'quiet': False,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✅ Saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")

def main():
    print(f"Saving files to: {DATA_DIR}")
    for url, name, start, end in DOWNLOAD_LIST:
        download_and_trim(url, name, start, end)

if __name__ == "__main__":
    main()