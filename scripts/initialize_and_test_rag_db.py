import os
import glob
import datetime
import re
import hashlib
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# --- CONFIGURATION ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
DATA_DIR = r"C:\Users\USER\Documents\meeting-asr\data\youtube_clips"
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

class SupabaseKnowledgeBase:
    def __init__(self):
        print("⚡ Connecting to Supabase...")
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.embedder = SentenceTransformer(MODEL_NAME, device='cpu')
        print("✅ Database Connected.")

    # --- 1. NEW DATE PARSER ---
    def parse_date_query(self, text):
        """Simple keyword matching to find dates (Hybrid Search Prep)"""
        today = datetime.date.today()
        text = text.lower()
        
        start_date = None
        end_date = None

        if "today" in text:
            start_date = today
            end_date = today
        elif "yesterday" in text:
            yesterday = today - datetime.timedelta(days=1)
            start_date = yesterday
            end_date = yesterday
        elif "last week" in text:
            start_date = today - datetime.timedelta(days=7)
            end_date = today
        elif "this month" in text:
            start_date = today.replace(day=1)
            end_date = today
            
        # Convert to strings if found
        s_str = start_date.strftime("%Y-%m-%d") if start_date else None
        e_str = end_date.strftime("%Y-%m-%d") if end_date else None
        
        return s_str, e_str

    def find_audio_match(self, text_filename):
        base_name = text_filename.split('.')[0]
        simple_name = base_name.split('_sample')[0] + "_sample" 
        search_pattern = os.path.join(DATA_DIR, f"*{simple_name}*.wav")
        matches = glob.glob(search_pattern)
        if not matches:
            search_pattern = os.path.join(DATA_DIR, f"{base_name}*.wav")
            matches = glob.glob(search_pattern)
        if matches: return matches[0] 
        return None

    def get_file_hash(self, file_path):
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def upload_audio_to_bucket(self, local_audio_path):
        if not local_audio_path or not os.path.exists(local_audio_path):
            return "N/A"
        file_name = os.path.basename(local_audio_path)
        print(f"   🔍 Calculating hash for: {file_name}...")
        file_hash = self.get_file_hash(local_audio_path)
        try:
            existing_files = self.supabase.storage.from_("meeting-audio").list("audio")
            for f in existing_files:
                if f['name'].startswith(file_hash):
                    print(f"   ♻️  Audio content match found: {f['name']} (Skipping upload)")
                    return self.supabase.storage.from_("meeting-audio").get_public_url(f"audio/{f['name']}")
        except Exception as e:
            print(f"   ⚠️ Could not check bucket list: {e}")

        unique_storage_path = f"audio/{file_hash}_{file_name}"
        print(f"   ☁️  Uploading NEW audio: {file_name}...")
        try:
            with open(local_audio_path, 'rb') as f:
                self.supabase.storage.from_("meeting-audio").upload(
                    path=unique_storage_path,
                    file=f,
                    file_options={"content-type": "audio/wav", "upsert": "false"}
                )
            return self.supabase.storage.from_("meeting-audio").get_public_url(unique_storage_path)
        except Exception as e:
            print(f"   ❌ Upload Failed: {e}")
            return "N/A"

    def extract_timestamp(self, text_chunk):
        match = re.search(r"\[(\d+\.?\d*)s", text_chunk)
        if match: return float(match.group(1))
        return 0.0

    def file_exists_in_db(self, filename, doc_type):
        try:
            response = self.supabase.table("meeting_chunks") \
                .select("id") \
                .eq("metadata->>filename", filename) \
                .eq("metadata->>type", doc_type) \
                .limit(1) \
                .execute()
            return len(response.data) > 0
        except:
            return False

    def add_file(self, file_path):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        except: return

        if filename.endswith(".md"):
            doc_type = "summary"
            chunk_size = 1000 
        else:
            doc_type = "transcript"
            chunk_size = 500

        if self.file_exists_in_db(filename, doc_type):
            print(f"⚠️  Skipping {filename} ({doc_type}): Already indexed.")
            return

        print(f"   ⬆️  Processing: {filename}...")
        local_audio_path = self.find_audio_match(filename)
        public_audio_url = "N/A"
        if local_audio_path:
            public_audio_url = self.upload_audio_to_bucket(local_audio_path)

        current_index = 0
        chunks_to_upload = []
        
        # NOTE: This uses today's date for indexing. 
        # If you want to test 'yesterday', change this line manually to a past date.
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")

        for i in range(0, len(content), chunk_size):
            chunk_text = content[i : i + chunk_size]
            if len(chunk_text.strip()) < 10: continue
            
            start_time = self.extract_timestamp(chunk_text)
            vector_values = self.embedder.encode(chunk_text).tolist()
            
            data_payload = {
                "content": chunk_text,
                "embedding": vector_values,
                "metadata": {
                    "filename": filename,
                    "type": doc_type,
                    "audio_path": public_audio_url, 
                    "chunk_index": current_index,
                    "start_time": start_time,
                    "date": today_str  # <--- Stored Date
                }
            }
            chunks_to_upload.append(data_payload)
            current_index += 1

        if chunks_to_upload:
            batch_size = 50
            for i in range(0, len(chunks_to_upload), batch_size):
                batch = chunks_to_upload[i : i + batch_size]
                self.supabase.table("meeting_chunks").insert(batch).execute()
            print(f"   ✅ Indexed {len(chunks_to_upload)} chunks.")

    def search(self, question):
        print(f"\n🐘 SUPABASE SEARCH: '{question}'")
        
        # 1. Extract Date Filters
        s_date, e_date = self.parse_date_query(question)
        if s_date:
            print(f"   🗓️  Date Filter Active: {s_date} to {e_date}")
        else:
            print("   🗓️  No date filter detected (Searching all time)")

        query_vector = self.embedder.encode(question).tolist()
        
        # 2. Pass filters to RPC
        response = self.supabase.rpc(
            "match_meetings", 
            {
                "query_embedding": query_vector,
                "match_threshold": 0.3,
                "match_count": 3,
                "filter_start_date": s_date, # New Param
                "filter_end_date": e_date    # New Param
            }
        ).execute()
        
        if not response.data:
            print("❌ No matches found.")
            return

        print("--- ANSWERS FOUND ---")
        for match in response.data:
            meta = match['metadata']
            timestamp = meta.get('start_time', 0.0)
            date_str = meta.get('date', 'Unknown')
            
            if timestamp > 0:
                time_str = str(datetime.timedelta(seconds=int(timestamp)))
                time_display = f"⏱️ {time_str}"
            else:
                time_display = "⏱️ Context"
            
            print(f"📄 {meta.get('filename')} [{date_str}]")
            print(f"{time_display} | 🎵 {meta.get('audio_path')}") 
            print(f"💬 \"{match['content'][:100].replace(chr(10), ' ')}...\"") 
            print("---")

if __name__ == "__main__":
    db = SupabaseKnowledgeBase()
    
    # 1. Index files
    text_files = glob.glob(os.path.join(DATA_DIR, "*.txt")) + glob.glob(os.path.join(DATA_DIR, "*.md"))
    print(f"\n📂 Found {len(text_files)} text files to process.")
    for f in text_files:
        db.add_file(f)

    print("\n" + "="*30)
    
    # 2. Test Searches
    # Since we just indexed them, they have TODAY'S date.
    
    # Test A: Should Find Results
    db.search("Apa pendapat PMX today?") 
    
    # Test B: Should Find NOTHING (because files are indexed as today, not yesterday)
    # This proves the filter is working if it returns 0 results.
    print("\nTesting 'Yesterday' filter (Should likely return nothing):")
    db.search("Apa pendapat PMX yesterday?")