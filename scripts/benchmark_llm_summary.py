import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
import time
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ================= CONFIGURATION =================

# 1. INPUT
TRANSCRIPT_PATH = Path(r"C:\Users\USER\Documents\meeting-asr\data\youtube_clips\mandarin_news_sample.lora_diarized.txt")

# 2. MODELS: Benchmarking All Candidates
MODELS_TO_TEST = [
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-2b-it",
]

# 3. AUTHENTICATION
HF_TOKEN = os.environ.get("HF_TOKEN")

# 4. SETTINGS
USE_WEB_SEARCH = True 
MAX_INPUT_CHARS = 8000       
MAX_WEB_CONTEXT_CHARS = 4000 

# 5. API KEYS
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

device = "cuda" if torch.cuda.is_available() else "cpu"

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
(3-5 sentences summarizing the main topic)

## 🔑 Key Points
- (Bullet point)
- (Bullet point)

## ✅ Action Items / Conclusions
(Only include if specific tasks or conclusions were mentioned)
"""

# ================= FUNCTIONS =================

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2) 

def load_model(model_id):
    print(f"\n[1/5] Loading {model_id} (4-bit)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            token=HF_TOKEN
        )
        return tokenizer, model
    except Exception as e:
        print(f"❌ Failed to load {model_id}: {e}")
        return None, None

def format_prompt_safe(tokenizer, system_msg, user_msg):
    """
    Robust wrapper that handles models (like Gemma) that crash 
    if you pass a 'system' role. It attempts standard formatting, 
    and falls back to merging if it fails.
    """
    # 1. Standard Format (System + User)
    try:
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
    except Exception:
        # 2. Fallback: Merge System into User (For Gemma/Phi)
        combined_msg = f"**SYSTEM INSTRUCTION:**\n{system_msg}\n\n**USER TASK:**\n{user_msg}"
        messages = [{"role": "user", "content": combined_msg}]
        return tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

def clean_transcript_for_query(text):
    """Removes [0.0s] SPEAKER_01: tags to prevent leakage into search queries"""
    # Regex removes: [anything] SPEAKER_xx:
    clean = re.sub(r'\[.*?\] SPEAKER_.*?:', '', text)
    # Regex removes simple names like: SPEAKER_01:
    clean = re.sub(r'SPEAKER_\d+:', '', clean)
    return clean.strip()

def generate_search_queries(text, tokenizer, model):
    print("\n[2/5] Extracting Key Sentences for Verification...")
    
    # Clean the speaker tags so the model doesn't copy them
    clean_text_preview = clean_transcript_for_query(text[:2000])
    
    prompt = f"""Read this transcript snippet carefully:
    "{clean_text_preview}..."
    
    **TASK:** Identify the 3 most fact-heavy sentences that contain specific names, places, or events.
    **CRITICAL:** Output the sentences EXACTLY as they appear in the transcript. Do not translate. Do not rephrase.
    
    Output exactly 3 lines.
    """
    
    try:
        inputs = format_prompt_safe(tokenizer, "You are a text extractor.", prompt)
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
        
        result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        queries = []
        for q in result.split('\n'):
            # Cleanup bullets/quotes
            clean_q = q.strip().replace('"', '').replace("'", "").replace("- ", "").replace("*", "")
            clean_q = re.sub(r'^[\d-]+\.\s*', '', clean_q) # Remove numbering "1. "
            
            # Filter out garbage
            if len(clean_q) > 10 and "SPEAKER" not in clean_q: 
                queries.append(clean_q)
        
        return queries[:3]
        
    except Exception as e:
        print(f"Extraction Error: {e}")
        return []

def scrape_url(url):
    try:
        import trafilatura
    except ImportError:
        return None

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            if text:
                return text[:1500] + "..." 
    except Exception:
        pass
    return None

def run_web_search(queries):
    if not queries: return None
    
    print(f"\n[3/5] Searching Web & Deep Scraping ({len(queries)} queries)...")
    raw_results = []
    
    for q in queries:
        print(f"   🔎 '{q[:50]}...'") 
        search_hits = []
        
        # 0. SERPAPI
        if SERPAPI_KEY and "YOUR_KEY" not in SERPAPI_KEY:
            try:
                from serpapi import GoogleSearch
                search = GoogleSearch({"q": q, "api_key": SERPAPI_KEY})
                results = search.get_dict().get("organic_results", [])
                if results:
                    print(f"      [SerpApi] Found {len(results)} results.")
                    for r in results[:2]:
                        search_hits.append({'title': r.get('title'), 'href': r.get('link'), 'body': r.get('snippet')})
            except Exception as e:
                print(f"      ⚠️ SerpApi failed: {e}")

        # 1. FALLBACKS (Google/DDG)
        if not search_hits:
            # ... (Existing fallback logic suppressed for brevity, assumed installed) ...
            pass

        # 3. PROCESS RESULTS
        if search_hits:
            for r in search_hits:
                title = r.get('title', 'No Title')
                link = r.get('href', r.get('url', ''))
                snippet = r.get('body', r.get('description', ''))
                
                print(f"         Found: {title[:50]}...")
                full_text = scrape_url(link)
                
                if full_text:
                    content_to_add = f"TITLE: {title}\nURL: {link}\nFULL TEXT: {full_text}\n"
                else:
                    content_to_add = f"TITLE: {title}\nURL: {link}\nSNIPPET: {snippet}\n"
                    
                raw_results.append(content_to_add)
        else:
            print("      -> ❌ No results found.")

    combined_results = "\n".join(raw_results)
    if len(combined_results) > MAX_WEB_CONTEXT_CHARS:
        print(f"   ⚠️ Web context truncated to {MAX_WEB_CONTEXT_CHARS} chars.")
        combined_results = combined_results[:MAX_WEB_CONTEXT_CHARS]
        
    return combined_results

def filter_context(transcript, raw_search_results, tokenizer, model):
    if not raw_search_results: return None 

    print("\n[4/5] Filtering Context for Relevance...")
    flush_memory()
    
    # Relaxed prompt to allow partial matches
    prompt = f"""RAW TRANSCRIPT TOPIC:
    "{transcript[:500]}..."
    
    SEARCH RESULTS:
    {raw_search_results}
    
    TASK: Do these search results provide ANY background info relevant to the transcript topic?
    - If YES, output the useful facts.
    - If NO, output "IRRELEVANT".
    """
    
    inputs = format_prompt_safe(tokenizer, "You are a helpful researcher.", prompt)
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=512, temperature=0.3)
    
    verified_context = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
    # Only block if it EXPLICITLY says Irrelevant AND is very short
    if "IRRELEVANT" in verified_context.upper() and len(verified_context) < 50:
        print("   ⚠️ Web results deemed irrelevant. Ignoring.")
        return None
        
    print("   ✅ Verified Context Extracted.")
    return verified_context

def generate_minutes(text, context, tokenizer, model):
    print("\n[5/5] Generating Final Minutes...")
    
    if len(text) > MAX_INPUT_CHARS:
        safe_text = text[:MAX_INPUT_CHARS]
    else:
        safe_text = text

    if context:
        user_prompt = f"""
        **VERIFIED CONTEXT:**
        {context}
        
        **TRANSCRIPT:**
        {safe_text}
        
        **INSTRUCTION:**
        Generate meeting minutes. Use the Verified Context to correct names/spellings in the transcript.
        """
    else:
        user_prompt = f"""
        **TRANSCRIPT:**
        {safe_text}
        
        **INSTRUCTION:**
        Generate meeting minutes based ONLY on the transcript.
        """
    
    inputs = format_prompt_safe(tokenizer, SYSTEM_PROMPT, user_prompt)
    
    flush_memory()
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=1024, temperature=0.7)
        
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

def metric_entity_density(summary):
    words = summary.split()
    if not words: return 0
    cap_count = sum(1 for w in words if w[0].isupper() and len(w) > 1)
    return round(cap_count / len(words), 3)

def metric_vagueness(summary):
    lazy_phrases = ["discusses", "talks about", "mentioned that", "various topics"]
    count = sum(1 for phrase in lazy_phrases if phrase in summary.lower())
    return count

def run_qag_evaluation(transcript, summary, tokenizer, model):
    print("\n[Evaluation] 🧪 Running QAG Factual Check...")
    
    # 1. Ask Model to generate questions based on TRANSCRIPT
    q_prompt = f"""Read this text:
    "{transcript[:2000]}..."
    
    Generate 3 very specific questions about names, numbers, or specific actions mentioned.
    Format:
    Q1: ...
    Q2: ...
    Q3: ...
    """
    try:
        inputs = format_prompt_safe(tokenizer, "You are a quiz generator.", q_prompt)
        outputs = model.generate(inputs, max_new_tokens=150)
        questions_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    except:
        return 0

    # 2. Ask Model to answer using SUMMARY
    a_prompt = f"""Use ONLY the summary below to answer the questions. 
    If the answer is NOT in the summary, reply EXACTLY "MISSING".
    
    SUMMARY:
    {summary}
    
    QUESTIONS:
    {questions_text}
    """
    try:
        inputs = format_prompt_safe(tokenizer, "You are a strict examiner.", a_prompt)
        outputs = model.generate(inputs, max_new_tokens=150)
        answers_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        # 3. Calculate Score
        missing_count = answers_text.upper().count("MISSING")
        score = max(0, 3 - missing_count)
        print(f"    👉 Score: {score}/3")
        return score
    except:
        return 0

# ================= MAIN LOOP =================

def main():
    if not TRANSCRIPT_PATH.exists():
        print(f"❌ File not found: {TRANSCRIPT_PATH}")
        return

    print(f"Reading: {TRANSCRIPT_PATH.name}")
    raw_text = TRANSCRIPT_PATH.read_text(encoding="utf-8")

    for model_id in MODELS_TO_TEST:
        print("\n" + "="*60)
        print(f"🧪 BENCHMARKING: {model_id}")
        print("="*60)
        
        flush_memory()
        
        tokenizer, model = load_model(model_id)
        if not model: continue

        web_context = None
        if USE_WEB_SEARCH:
            queries = generate_search_queries(raw_text, tokenizer, model)
            if queries:
                print(f"   (Queries: {queries})")
                raw_results = run_web_search(queries)
                if raw_results:
                    web_context = filter_context(raw_text, raw_results, tokenizer, model)
            else:
                print("   (Model detected PRIVATE meeting or no queries generated)")

        start_time = time.time()
        minutes = generate_minutes(raw_text, web_context, tokenizer, model)
        elapsed = time.time() - start_time
        
        short_name = model_id.split("/")[-1]
        out_path = TRANSCRIPT_PATH.with_suffix(f".{short_name}.FINAL.md")
        out_path.write_text(minutes, encoding="utf-8")
        
        print("\n" + "-"*40)
        print(f"✅ Finished {short_name} in {elapsed:.1f}s")
        print(f"📄 Saved to: {out_path}")
        print("-"*40)
        
        qag_score = run_qag_evaluation(raw_text, minutes, tokenizer, model)
        density = metric_entity_density(minutes)
        
        print(f"📊 REPORT: QAG Score={qag_score}/3 | Density={density}")
        
        # Save to file
        out_path = TRANSCRIPT_PATH.with_suffix(f".{short_name}.FINAL.md")
        out_path.write_text(f"{minutes}\n\n---\nSCORE: {qag_score}/3", encoding="utf-8")
        
        del model
        del tokenizer
        flush_memory()

if __name__ == "__main__":
    main()
    
    