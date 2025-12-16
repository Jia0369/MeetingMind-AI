import streamlit as st
import os
import gc
import re
import torch
import librosa
import uuid
import datetime
import warnings
import langid 
import hashlib

# --- 0. SUPPRESS WARNINGS (Clean Console) ---
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. MEMORY OPTIMIZATION SETUP ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 2. NUCLEAR FIX FOR TORCHCODEC ---
import transformers.utils.import_utils
transformers.utils.import_utils.is_torchcodec_available = lambda: False
try:
    import transformers.pipelines.automatic_speech_recognition
    transformers.pipelines.automatic_speech_recognition.is_torchcodec_available = lambda: False
except (ImportError, AttributeError):
    pass

# --- 3. IMPORTS ---
from pyannote.audio import Pipeline
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    pipeline,
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

try:
    import trafilatura
except ImportError:
    trafilatura = None

# ================= BACKEND CONFIGURATION =================

# 🔒 HIDDEN PATHS & KEYS
LORA_CHECKPOINT_PATH = r"C:\Users\USER\Documents\meeting-asr\models\whisper-small-lora-trilingual"
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")
# SUPABASE CONFIG
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# MODEL SETTINGS
DEFAULT_WHISPER_BASE = "openai/whisper-small"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct" 
HF_TOKEN = os.environ.get("HF_TOKEN")
# LIMITS
MAX_INPUT_CHARS = 8000        
MAX_WEB_CONTEXT_CHARS = 4000

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MEMORY MANAGEMENT =================

def flush_memory():
    """Aggressively clears VRAM and RAM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ================= DATABASE & RAG CLASS =================

@st.cache_resource
def get_db_client():
    """Initializes Supabase and Embedder once"""
    return SupabaseKnowledgeBase()

class SupabaseKnowledgeBase:
    def __init__(self):
        print("⚡ Connecting to Supabase...")
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu") 
        print("✅ Database Connected.")

    # 🆕 NEW: Check if file exists
    def check_if_hash_exists(self, file_hash):
        try:
            response = self.supabase.table("meeting_chunks") \
                .select("metadata") \
                .contains("metadata", {"file_hash": file_hash}) \
                .limit(1) \
                .execute()
            if response.data:
                meta = response.data[0]['metadata']
                return meta.get('session_id'), meta.get('filename')
            return None, None
        except Exception as e:
            return None, None

    def index_live_session(self, transcript_text, summary_text, audio_source, file_hash, reuse_audio_url=None, custom_filename=None):
        if reuse_audio_url:
            public_audio_url = reuse_audio_url
        else:
            public_audio_url = self.upload_audio_to_bucket(audio_source)

        chunks_to_upload = []
        current_index = 0
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        final_filename = custom_filename if custom_filename else f"Session_{date_str}"
        session_id = str(uuid.uuid4())

        # Transcript Chunks
        for i in range(0, len(transcript_text), 500):
            chunk_text = transcript_text[i : i + 500]
            if len(chunk_text.strip()) < 10: continue
            chunks_to_upload.append({
                "content": chunk_text,
                "embedding": self.embedder.encode(chunk_text).tolist(),
                "metadata": {
                    "filename": final_filename, "type": "transcript", "audio_path": public_audio_url, 
                    "chunk_index": current_index, "start_time": self.extract_timestamp(chunk_text),
                    "date": date_str, "session_id": session_id,
                    "file_hash": file_hash  # <--- Saving Hash
                }
            })
            current_index += 1

        # Summary Chunks
        for i in range(0, len(summary_text), 1000):
            chunk_text = summary_text[i : i + 1000]
            if len(chunk_text.strip()) < 10: continue
            chunks_to_upload.append({
                "content": chunk_text,
                "embedding": self.embedder.encode(chunk_text).tolist(),
                "metadata": {
                    "filename": final_filename, "type": "summary", "audio_path": public_audio_url,
                    "start_time": 0.0, "date": date_str, "session_id": session_id,
                    "file_hash": file_hash  # <--- Saving Hash
                }
            })

        if chunks_to_upload:
            for i in range(0, len(chunks_to_upload), 50):
                batch = chunks_to_upload[i : i + 50]
                try: self.supabase.table("meeting_chunks").insert(batch).execute()
                except: pass
            return True
        return False

    # ... (Keep delete_session, rename_session, get_history, load_session, search as they were) ...
    def delete_session(self, session_id):
        try:
            self.supabase.table("meeting_chunks").delete().eq("metadata->>session_id", session_id).execute()
            return True
        except: return False

    def rename_session(self, session_id, new_name):
        try:
            self.supabase.rpc("rename_session_metadata", {"p_session_id": session_id, "p_new_name": new_name}).execute()
            return True
        except: return False

    def get_history(self):
        try:
            response = self.supabase.table("meeting_chunks").select("metadata, created_at").order("created_at", desc=True).limit(2000).execute()
            seen, history = set(), []
            for item in response.data:
                meta = item.get('metadata', {})
                sid = meta.get('session_id')
                if sid and sid not in seen:
                    seen.add(sid)
                    time_str = ""
                    try:
                        clean_iso = item.get('created_at', '').replace('Z', '+00:00')
                        time_str = datetime.datetime.fromisoformat(clean_iso).astimezone().strftime("%H:%M")
                    except: pass
                    history.append({"id": sid, "date": meta.get('date', 'Unknown'), "time": time_str, "name": meta.get('filename', 'Unknown')})
            return history
        except: return []

    def load_session(self, session_id):
        try:
            response = self.supabase.table("meeting_chunks").select("content, metadata").contains("metadata", f'{{"session_id": "{session_id}"}}').execute()
            data = response.data
            if not data: return None, None, None, None
            first_meta = data[0]['metadata']
            trans_chunks = sorted([d for d in data if d['metadata'].get('type') == 'transcript'], key=lambda x: x['metadata'].get('chunk_index', 0))
            full_trans = "".join([c['content'] for c in trans_chunks])
            sum_chunks = [d for d in data if d['metadata'].get('type') == 'summary']
            full_summ = "".join([c['content'] for c in sum_chunks])
            return full_trans, full_summ, first_meta.get('audio_path'), first_meta.get('filename')
        except: return None, None, None, None

    def search(self, question):
        s_date, e_date = self.parse_date_query(question)
        query_vector = self.embedder.encode(question).tolist()
        try:
            return self.supabase.rpc("match_meetings", {
                "query_embedding": query_vector, "match_threshold": 0.5, "match_count": 10,
                "filter_start_date": s_date, "filter_end_date": e_date
            }).execute().data
        except: return []

    def parse_date_query(self, text):
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
        s_str = start_date.strftime("%Y-%m-%d") if start_date else None
        e_str = end_date.strftime("%Y-%m-%d") if end_date else None
        return s_str, e_str

    def extract_timestamp(self, text_chunk):
        match = re.search(r"\[(\d+\.?\d*)s", text_chunk)
        if match: return float(match.group(1))
        return 0.0

    def upload_audio_to_bucket(self, local_audio_path):
        if not local_audio_path or not os.path.exists(local_audio_path): return "N/A"
        file_ext = os.path.splitext(local_audio_path)[1]
        unique_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}{file_ext}"
        storage_path = f"audio/{unique_name}"
        try:
            with open(local_audio_path, 'rb') as f:
                self.supabase.storage.from_("meeting-audio").upload(
                    path=storage_path, file=f, file_options={"content-type": "audio/wav", "upsert": "false"}
                )
            return self.supabase.storage.from_("meeting-audio").get_public_url(storage_path)
        except Exception as e:
            print(f"    ❌ Upload Failed: {e}")
            return "N/A"
        
# ================= PROMPTS =================

SYSTEM_PROMPT = """You are an expert content summarizer.
Your goal is to summarize the provided transcript clearly and concisely.

**CORE RESPONSIBILITY:**
- Summarize the content accurately based ONLY on the provided text.
- **NEUTRALITY:** Do not inject personal opinions.
- **NO HALLUCINATION:** Do not invent metadata (like Date, Location, Attendees) if they are not explicitly stated in the text.

**OUTPUT FORMAT:**
# 📝 Summary Report

## 🎯 Executive Summary
(Write sentences summarizing the main topic)

## 🔑 Key Points
- (Bullet point)
- (Bullet point)

## ✅ Action Items / Conclusions
(Only include if specific tasks or conclusions were mentioned)
"""

# ================= HELPER FUNCTIONS =================

def clean_text(text):
    allowed = re.compile(r'[^\u0000-\u007F\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+')
    return allowed.sub('', text).strip()

def clean_transcript_for_query(text):
    clean = re.sub(r'\[.*?\] SPEAKER_.*?:', '', text)
    clean = re.sub(r'SPEAKER_\d+:', '', clean)
    return clean.strip()

def scrape_url(url):
    if not trafilatura: return None
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text: return text[:1500] + "..." 
    except: pass
    return None

def mark_unsaved():
    st.session_state.is_saved = False

def reset_application():
    st.session_state.transcript = None
    st.session_state.original_transcript = None
    st.session_state.summary = None
    st.session_state.original_summary = None
    st.session_state.messages = []
    st.session_state.is_saved = False 
    st.session_state.uploader_key += 1
    st.session_state.speaker_mapping = {}
    
    if "current_audio_url" in st.session_state: del st.session_state.current_audio_url
    if "current_filename" in st.session_state: del st.session_state.current_filename
    
    if "summary_edit_content" in st.session_state: del st.session_state.summary_edit_content
    if "transcript_area" in st.session_state: del st.session_state.transcript_area
    if "delete_confirm_id" in st.session_state: del st.session_state.delete_confirm_id

    if os.path.exists("temp_input.wav"):
        try: os.remove("temp_input.wav")
        except: pass
    
    flush_memory()
    st.rerun()

# ================= INTELLIGENCE ENGINES =================

def load_llm():
    flush_memory()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_LLM_MODEL, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_LLM_MODEL, quantization_config=bnb_config, device_map=device, token=HF_TOKEN
    )
    return model, tokenizer

def format_prompt_safe(tokenizer, system_msg, user_msg):
    try:
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
    except:
        combined = f"**SYSTEM:** {system_msg}\n\n**USER:** {user_msg}"
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": combined}], 
            tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

# ================= PIPELINE STAGES =================

def run_diarization(audio_path, status_container):
    status_container.info("🕵️‍♂️ Stage 1: Running Diarization...")
    flush_memory()
    dia_pipeline = None
    try:
        dia_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
        ).to(torch.device("cpu"))
        
        diarization = dia_pipeline(str(audio_path))
        segments = [{"start": t.start, "end": t.end, "speaker": s} for t, _, s in diarization.itertracks(yield_label=True)]
        return segments
    except Exception as e:
        status_container.error(f"Diarization Failed: {e}")
        return []
    finally:
        del dia_pipeline
        flush_memory()

def run_transcription(audio_path, segments, status_container):
    status_container.info("🎤 Stage 2: Transcribing (Auto-Detect)...")
    flush_memory()
    model = None
    asr_pipe = None
    try:
        y, sr = librosa.load(str(audio_path), sr=16000)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            DEFAULT_WHISPER_BASE, low_cpu_mem_usage=True, torch_dtype=torch.float16
        )
        try:
            model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT_PATH).merge_and_unload().eval()
        except:
            model = base_model
        model = model.to(device)
        processor = WhisperProcessor.from_pretrained(DEFAULT_WHISPER_BASE)
        asr_pipe = pipeline(
            "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor, return_timestamps=True, torch_dtype=torch.float16
        )
        
        transcript_lines = []
        prog_bar = status_container.progress(0)
        for i, seg in enumerate(segments):
            if (seg["end"] - seg["start"]) < 0.5: continue
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            audio_slice = y[start_sample:end_sample]
            try:
                gen_kwargs = {
                    "task": "transcribe", "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3, "condition_on_prev_tokens": False
                }
                out = asr_pipe(audio_slice, batch_size=1, generate_kwargs=gen_kwargs)
                raw_text = out["text"].strip()
                final_lang = "auto"
                if raw_text:
                    detected_lang, confidence = langid.classify(raw_text)
                    if detected_lang == 'id': detected_lang = 'ms'
                    final_lang = detected_lang
                txt = clean_text(raw_text)
                if len(txt) > 1:
                    transcript_lines.append(f"[{seg['start']:.1f}s] [{final_lang}] {seg['speaker']}: {txt}")
            except: pass
            prog_bar.progress((i+1)/len(segments))
            if i % 10 == 0: flush_memory()
        return "\n".join(transcript_lines)
    finally:
        del model
        del asr_pipe
        del y
        flush_memory()

def run_web_verified_summary(transcript, status_container):
    status_container.info("🧠 Stage 3: Verifying Facts with Web Search...")
    flush_memory()
    model, tokenizer = load_llm()
    try:
        status_container.write("🔎 Analyzing transcript for search terms...")
        clean_preview = clean_transcript_for_query(transcript[:2000])
        query_prompt = f"""Read this transcript snippet:
        "{clean_preview}..."
        **TASK:** Identify the 3 most fact-heavy sentences. Output exactly 3 lines."""
        inputs = format_prompt_safe(tokenizer, "You are a research assistant.", query_prompt)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=100, temperature=0.1)
        raw_queries = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        queries = []
        for q in raw_queries.split('\n'):
            clean_q = q.strip().replace('"', '').replace("'", "").replace("- ", "").replace("*", "")
            if len(clean_q) > 10 and "SPEAKER" not in clean_q:
                queries.append(clean_q)
        queries = queries[:3]
        
        web_context_candidates = []
        if queries:
            status_container.write(f"🌐 Searching Web ({len(queries)} queries)...")
            for q in queries:
                search_hits = []
                if SERPAPI_KEY:
                    try:
                        from serpapi import GoogleSearch
                        search = GoogleSearch({"q": q, "api_key": SERPAPI_KEY})
                        results = search.get_dict().get("organic_results", [])
                        for r in results[:2]:
                            search_hits.append({'title': r.get('title'), 'href': r.get('link'), 'body': r.get('snippet')})
                    except Exception as e: print(f"SerpApi Error: {e}")
                if not search_hits:
                    try:
                        from duckduckgo_search import DDGS
                        hits = DDGS().text(q, max_results=2, backend='lite')
                        if hits: 
                            for h in hits: search_hits.append({'title': h['title'], 'href': h['href'], 'body': h['body']})
                    except: pass
                if search_hits:
                    for r in search_hits:
                        full_text = scrape_url(r['href'])
                        if full_text: web_context_candidates.append(f"TITLE: {r['title']}\nURL: {r['href']}\nFULL TEXT: {full_text}\n")
                        else: web_context_candidates.append(f"TITLE: {r['title']}\nURL: {r['href']}\nSNIPPET: {r['body']}\n")

        raw_web_context = "\n".join(web_context_candidates)[:MAX_WEB_CONTEXT_CHARS]
        verified_context = "No verified context found."
        if raw_web_context:
            status_container.write("🕵️ Filtering search results...")
            filter_prompt = f"""RAW TOPIC: "{clean_transcript_for_query(transcript[:500])}..."
            SEARCH RESULTS: {raw_web_context}
            TASK: Do results provide background info? IF YES, output facts. IF NO, output "IRRELEVANT"."""
            f_inputs = format_prompt_safe(tokenizer, "You are a helpful researcher.", filter_prompt)
            with torch.no_grad():
                f_outputs = model.generate(f_inputs, max_new_tokens=512, temperature=0.3)
            filtered_text = tokenizer.decode(f_outputs[0][len(f_inputs[0]):], skip_special_tokens=True)
            if "IRRELEVANT" not in filtered_text.upper():
                verified_context = filtered_text
                status_container.success("✅ Relevant Web Context Extracted")
            else:
                status_container.warning("⚠️ Web results deemed irrelevant.")

        status_container.write("📝 Drafting Final Summary...")
        final_prompt = f"""**VERIFIED CONTEXT:** {verified_context}
        **TRANSCRIPT:** {transcript[:MAX_INPUT_CHARS]}
        **INSTRUCTION:** Generate a structured summary report. Use Context to correct spellings."""
        inputs = format_prompt_safe(tokenizer, SYSTEM_PROMPT, final_prompt)
        full_text_output = model.generate(inputs, max_new_tokens=1024, temperature=0.7)
        decoded_text = tokenizer.decode(full_text_output[0][len(inputs[0]):], skip_special_tokens=True)
        return decoded_text
    except Exception as e:
        status_container.error(f"Summary Error: {e}")
        return "Summary generation failed."
    finally:
        del model
        del tokenizer
        flush_memory()
        
def get_file_hash(uploaded_file):
    """Calculates SHA256 hash of the uploaded file."""
    # Move pointer to start
    uploaded_file.seek(0)
    # Read file and calculate hash
    file_hash = hashlib.sha256(uploaded_file.read()).hexdigest()
    # CRITICAL: Reset pointer to start so other functions can read it later
    uploaded_file.seek(0)
    return file_hash

def get_rag_context(prompt, current_transcript):
    """
    Phase 1: Search the database and prepare context + sources.
    Returns: context_block (str), source_list (list of dicts)
    """
    db = get_db_client()
    rag_context = ""
    source_list = []
    
    # 1. Search Vector DB
    try:
        matches = db.search(prompt)
        if matches:
            rag_context = "\n--- DATABASE CONTEXT ---\n"
            for m in matches:
                meta = m['metadata']
                # Store full metadata for UI rendering
                source_list.append(meta)
                
                # Build text context for LLM
                fname = meta.get('filename', 'Unknown')
                rag_context += f"- [{fname}]: {m['content'][:200]}...\n"
    except Exception as e:
        print(f"RAG Error: {e}")

    # 2. Build Context Block for LLM
    safe_current_context = current_transcript[:4000] if current_transcript else ""
    context_block = f"CURRENT TRANSCRIPT:\n{safe_current_context}\n\n{rag_context}"
    
    return context_block, source_list

def stream_llm_response(prompt, context_block):
    """
    Phase 2: Just generate the text response using the context.
    """
    model, tokenizer = load_llm()
    try:
        # Build Messages
        messages = [{"role": "system", "content": f"You are a helpful assistant. Use this context to answer:\n{context_block}"}]
        
        # Add History
        for msg in st.session_state.messages[-4:]: 
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": prompt})

        # Generate
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(device)
        
        # Streamer (Simulated via simple generate for now to keep your logic consistent)
        outputs = model.generate(inputs, max_new_tokens=1024)
        response_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        return response_text
    finally:
        del model
        del tokenizer
        flush_memory()                
# ================= MAIN UI =================

st.set_page_config(page_title="MeetingMind", page_icon="🧠", layout="wide")

if "transcript" not in st.session_state: st.session_state.transcript = None
if "original_transcript" not in st.session_state: st.session_state.original_transcript = None
if "summary" not in st.session_state: st.session_state.summary = None
if "original_summary" not in st.session_state: st.session_state.original_summary = None
if "messages" not in st.session_state: st.session_state.messages = []
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0 
if "is_saved" not in st.session_state: st.session_state.is_saved = False
if "speaker_mapping" not in st.session_state: st.session_state.speaker_mapping = {}

if "current_audio_url" not in st.session_state: st.session_state.current_audio_url = None
if "current_filename" not in st.session_state: st.session_state.current_filename = None
if "delete_confirm_id" not in st.session_state: st.session_state.delete_confirm_id = None 

st.title("🧠 MeetingMind")
st.markdown("### *AI-Powered Transcription & Meeting Memory Application*")

# --- SIDEBAR (INPUT & HISTORY) ---
with st.sidebar:
    st.header("1. Actions")
    st.button("🗑️ Reset / New Meeting", on_click=reset_application, use_container_width=True)
    st.write("---")
    
    # --- HISTORY PANEL ---
    with st.expander("📜 Meeting History", expanded=True):
        st.caption("Load, Rename, or Delete past sessions.")
        try:
            db_client = get_db_client()
            history_list = db_client.get_history()
            
            if history_list:
                options = {}
                for h in history_list:
                    date_str = h.get('date', 'Unknown Date')
                    time_str = h.get('time', '') 
                    name_str = h.get('name', 'Unnamed Session')
                    label = f"{date_str} {f'@ {time_str}' if time_str else ''} | {name_str}"
                    options[label] = h['id']
                    # Store raw name for pre-filling the rename box
                    options[f"{label}_RAW_NAME"] = name_str
                
                selected_option = st.selectbox("Select Session:", options=[k for k in options.keys() if "_RAW_NAME" not in k])
                sel_id = options[selected_option]
                current_name = options[f"{selected_option}_RAW_NAME"]

                # --- ACTIONS ---
                st.write("---")
                
                # Rename Section
                new_name_input = st.text_input("Edit Name:", value=current_name, key=f"rename_{sel_id}")
                if st.button("✏️ Rename", use_container_width=True):
                    if new_name_input != current_name:
                        with st.spinner("Updating database..."):
                            success = db_client.rename_session(sel_id, new_name_input)
                            if success:
                                st.success("Renamed!")
                                st.rerun()
                            else:
                                st.error("Failed to rename.")
                
                col_load, col_del = st.columns([3, 1])
                
                if col_load.button("📂 Load Session", use_container_width=True):
                    with st.spinner("Fetching..."):
                        h_trans, h_summ, h_audio, h_fname = db_client.load_session(sel_id)
                        if h_trans and h_summ:
                            st.session_state.transcript = h_trans
                            st.session_state.original_transcript = h_trans
                            st.session_state.summary = h_summ
                            st.session_state.original_summary = h_summ
                            st.session_state.summary_edit_content = h_summ
                            st.session_state.transcript_area = h_trans
                            st.session_state.current_audio_url = h_audio
                            st.session_state.current_filename = h_fname
                            st.session_state.is_saved = True 
                            st.session_state.messages = [] 
                            st.success("Loaded!")
                            st.rerun()

                if col_del.button("❌", use_container_width=True):
                    st.session_state.delete_confirm_id = sel_id

                # Delete Confirmation Logic (Same as before)
                if st.session_state.delete_confirm_id == sel_id:
                    st.warning("⚠️ Delete this session?")
                    col_conf_yes, col_conf_no = st.columns(2)
                    if col_conf_yes.button("✅ Yes", key="conf_yes"):
                        with st.spinner("Deleting..."):
                            success = db_client.delete_session(sel_id)
                            if success:
                                st.session_state.delete_confirm_id = None 
                                reset_application()
                            else:
                                st.error("Delete failed.")
                    if col_conf_no.button("Cancel", key="conf_no"):
                        st.session_state.delete_confirm_id = None
                        st.rerun()

            else:
                st.info("No history found.")
        except Exception as e:
            st.error(f"Connection Error: {e}")

        st.write("---")
        st.header("2. New Input")
        tab1, tab2 = st.tabs(["📂 File", "🎙️ Record"])
        audio_file = None
        with tab1:
            u_file = st.file_uploader("Upload", type=["wav", "mp3", "m4a"], key=f"u_file_{st.session_state.uploader_key}")
            if u_file: audio_file = u_file
        with tab2:
            r_file = st.audio_input("Record", key=f"r_file_{st.session_state.uploader_key}")
            if r_file: audio_file = r_file
        
        # 🔒 DUPLICATE CHECK
        current_file_hash = None
        existing_session_id = None
        if audio_file:
            current_file_hash = get_file_hash(audio_file)
            db_client = get_db_client()
            existing_session_id, existing_name = db_client.check_if_hash_exists(current_file_hash)
            
            if existing_session_id:
                st.warning(f"⚠️ **Duplicate Detected!** This audio is already saved as: `{existing_name}`")
                if st.button("📂 Load Existing Session"):
                    with st.spinner("Loading..."):
                        h_trans, h_summ, h_audio, h_fname = db_client.load_session(existing_session_id)
                        st.session_state.transcript = h_trans
                        st.session_state.original_transcript = h_trans
                        st.session_state.summary = h_summ
                        st.session_state.original_summary = h_summ
                        st.session_state.summary_edit_content = h_summ
                        st.session_state.transcript_area = h_trans
                        st.session_state.current_audio_url = h_audio
                        st.session_state.current_filename = h_fname
                        st.session_state.is_saved = True
                        st.session_state.messages = []
                        st.rerun()
    
        st.divider()
        # Disable processing if duplicate found
        process_btn = st.button("🚀 Process New Meeting", type="primary", disabled=(not audio_file or existing_session_id is not None))
    
    # --- PROCESSING ---
    if process_btn and audio_file and not existing_session_id:
        if os.path.exists("temp_input.wav"): os.remove("temp_input.wav")
        with open("temp_input.wav", "wb") as f: f.write(audio_file.read())
        status = st.status("Running AI Pipeline...", expanded=True)
        st.session_state.is_saved = False
        st.session_state.speaker_mapping = {} 
        
        # Save hash for later
        st.session_state.current_file_hash = current_file_hash
        
        # 1. Diarize
        segments = run_diarization("temp_input.wav", status)
        if not segments:
            status.warning("⚠️ Diarization found no speakers. Fallback...")
            segments = [{"start": 0.0, "end": 36000.0, "speaker": "Speaker"}]
        
        # 2. Transcribe
        transcript = run_transcription("temp_input.wav", segments, status)
        st.session_state.transcript = transcript
        st.session_state.original_transcript = transcript 
        if "transcript_area" in st.session_state: st.session_state.transcript_area = transcript
        
        # 3. Summary
        summary = run_web_verified_summary(transcript, status)
        st.session_state.summary = summary
        st.session_state.original_summary = summary 
        st.session_state.summary_edit_content = summary
        
        # Reset versioning vars for fresh file
        st.session_state.current_audio_url = None
        st.session_state.current_filename = None
        
        status.write("✅ Processing Complete.")
        status.update(label="✅ Complete", state="complete", expanded=False)

# --- RESULTS DASHBOARD (ALWAYS VISIBLE) ---
tab_transcript, tab_summary, tab_chat = st.tabs(["📄 Transcript", "📝 Verified Minutes", "💬 AI Chat (RAG)"])

with tab_transcript:
    if st.session_state.transcript:
        if st.session_state.original_transcript:
            speaker_pattern = re.compile(r"\bSPEAKER_\d+\b")
            found_speakers = sorted(list(set(speaker_pattern.findall(st.session_state.original_transcript))))
            if found_speakers:
                with st.expander("✍️ Rename Speakers", expanded=False):
                    with st.form("speaker_rename_form"):
                        name_map = {}
                        cols = st.columns(2)
                        for i, spk in enumerate(found_speakers):
                            with cols[i % 2]:
                                current_val = st.session_state.speaker_mapping.get(spk, spk)
                                name_map[spk] = st.text_input(f"Rename {spk} to:", value=current_val)
                        if st.form_submit_button("Apply Changes"):
                            st.session_state.speaker_mapping.update(name_map)
                            new_transcript = st.session_state.original_transcript
                            new_summary = st.session_state.original_summary
                            for spk_tag, mapped_name in st.session_state.speaker_mapping.items():
                                if spk_tag != mapped_name:
                                    new_transcript = new_transcript.replace(spk_tag, mapped_name)
                                    if new_summary: new_summary = new_summary.replace(spk_tag, mapped_name)
                            st.session_state.transcript = new_transcript
                            st.session_state.summary = new_summary
                            st.session_state.summary_edit_content = new_summary
                            if "transcript_area" in st.session_state: st.session_state.transcript_area = new_transcript
                            st.rerun()

        st.text_area("Full Text", st.session_state.transcript, height=500, key="transcript_area")
        st.download_button("Download Transcript", st.session_state.transcript, "transcript.txt")
    else:
        st.info("ℹ️ No meeting loaded. Upload a file or load from History to view transcript.")
    
with tab_summary:
    if st.session_state.summary:
        st.markdown("### Review & Edit")
        if "summary_edit_content" not in st.session_state:
            st.session_state.summary_edit_content = st.session_state.summary

        edited_summary = st.text_area(
            "Make any corrections before saving:",
            key="summary_edit_content",
            height=400,
            on_change=mark_unsaved 
        )
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if not st.session_state.is_saved:
                btn_label = "💾 Save New Version" if st.session_state.current_audio_url else "💾 Approve & Save"
                
                if st.button(btn_label, type="primary"):
                    with st.spinner("Saving to Knowledge Base..."):
                        try:
                            db = get_db_client()
                            version_name = None
                            if st.session_state.current_filename:
                                timestamp = datetime.datetime.now().strftime("%H:%M")
                                base = st.session_state.current_filename.split(" (Edited")[0]
                                version_name = f"{base} (Edited {timestamp})"
                            
                            success = db.index_live_session(
                                st.session_state.transcript, 
                                st.session_state.summary_edit_content, 
                                "temp_input.wav", 
                                file_hash=st.session_state.get("current_file_hash", "legacy_no_hash"), # <--- PASS HASH HERE
                                reuse_audio_url=st.session_state.current_audio_url,
                                custom_filename=version_name
                            )
                            if success:
                                st.session_state.is_saved = True
                                st.success(f"✅ Saved as {version_name if version_name else 'New Session'}!")
                                st.rerun()
                            else:
                                st.error("Failed to save.")
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.success("✅ Saved to Knowledge Base.")
                
        with col2:
            st.download_button("Download Minutes", st.session_state.summary_edit_content, "minutes.md")
    else:
        st.info("ℹ️ No summary available.")
        
with tab_chat:
    st.caption("Ask questions about this meeting or previous meetings (via Knowledge Base).")
    
    # 1. Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If this message has audio sources saved in history, render them
            if "sources" in msg and msg["sources"]:
                with st.expander("🎧 Play Audio Sources", expanded=False):
                    for src in msg["sources"]:
                        # Display format: Filename (Time)
                        lbl = f"📂 {src['filename']} @ {src['time_str']}"
                        st.caption(lbl)
                        # Render Audio Player jumped to specific time
                        if src.get('audio_url') and src['audio_url'] != "N/A":
                            st.audio(src['audio_url'], start_time=int(src['seconds']))
                        else:
                            st.warning("Audio file not found for this segment.")

    # 2. Handle New Input
    if prompt := st.chat_input("Ex: What were the deadlines?"):
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking & Searching..."):
                flush_memory()
                
                # A. Get Context & Sources
                context_block, raw_sources = get_rag_context(prompt, st.session_state.transcript)
                
                # B. Generate Text Response
                full_response = stream_llm_response(prompt, context_block)
                st.markdown(full_response)
                
                # C. Process Sources for UI
                clean_sources = []
                if raw_sources:
                    with st.expander("🎧 Play Audio Sources", expanded=True):
                        seen_timestamps = set()
                        
                        for meta in raw_sources:
                            fname = meta.get('filename', 'Unknown')
                            seconds = float(meta.get('start_time', 0))
                            audio_url = meta.get('audio_path')
                            
                            # Create a unique key to prevent duplicate players for the exact same second
                            unique_key = f"{fname}_{int(seconds)}"
                            if unique_key in seen_timestamps:
                                continue
                            seen_timestamps.add(unique_key)
                            
                            # Format nicely
                            time_str = str(datetime.timedelta(seconds=int(seconds)))
                            
                            st.markdown(f"**📄 {fname}** at `{time_str}`")
                            
                            # RENDER AUDIO PLAYER
                            if audio_url and audio_url != "N/A":
                                st.audio(audio_url, start_time=int(seconds))
                            else:
                                st.caption("*(Audio not available for this legacy meeting)*")
                                
                            # Save clean data for history
                            clean_sources.append({
                                "filename": fname,
                                "time_str": time_str,
                                "seconds": seconds,
                                "audio_url": audio_url
                            })

            # D. Save Assistant Message to History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": clean_sources # Store sources to render them again later
            })