# 🧠 MeetingMind-AI

**AI application that transcribes, diarizes, and summarizes multilingual meeting audio, storing the results in a vector-based knowledge base for real-time, context-aware Q&A via Streamlit.**

MeetingMind-AI is a comprehensive Final Year Project (FYP) built to process multilingual and code-switching meeting audio (supporting English, Mandarin, and Malay). The system leverages an efficient LoRA-tuned Whisper model for transcription and Pyannote for speaker diarization. The resulting minutes are saved to a Supabase Vector Database to enable Q&A and knowledge retrieval via a secure Streamlit web interface.

## ✨ Project Components

| Component | Technology / Goal |
| :--- | :--- |
| **ASR & Diarization** | LoRA-tuned `whisper-small` for transcription; `pyannote/speaker-diarization-3.1` for speaker separation. |
| **Minutes Generation** | Qwen2.5-3B-Instruct LLM with built-in **Web Search verification** (SerpAPI/DuckDuckGo) to ensure factual accuracy and correct entity spelling. |
| **Knowledge Base (RAG)** | **Supabase Vector Database** for storing historical meeting data. Enables context-aware Q&A across multiple past sessions. |
| **User Interface** | Secure, containerized Streamlit application (`streamlit_app.py`) for end-to-end workflow management. |

## ⚙️ Setup and Installation (For Examiners/Markers)

### 1. Prerequisites

* **Python 3.10+**
* **FFmpeg** (Must be installed and accessible via the system PATH for audio processing).

### 2. Clone Repository & Install Dependencies

Clone the project and install all required packages listed in `requirements.txt`:

```bash
git clone [https://github.com/Jia0369/MeetingMind-AI.git](https://github.com/Jia0369/MeetingMind-AI.git)
cd MeetingMind-AI
python -m venv venv
.\venv\Scripts\activate  # Windows
# or source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
