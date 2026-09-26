# Audio RAG & Diarization Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) pipeline designed for audio processing, speaker diarization, chunking, database indexing, and evaluation.

---

## Architecture & Project Structure

```text
├── data/
│   ├── audio/                     # Raw input audio files (.mp3, .wav)
│   └── transcript_diarized/       # Generated diarization JSON transcripts
│       └── chunked/               # Chunked transcript JSON files
├── db_setup/
│   └── postgress_setup.yaml       # Docker Compose setup for PostgreSQL + pgvector
├── run_pipeline.py                # Cross-platform master execution script
├── requirements.txt               # Python package dependencies
├── transripts_diarization.py      # Transcribes & diarizes audio (Whisper + PyAnnote)
├── transcript_chunking.py         # Chunks diarized transcripts for search indexing
├── ingest.py                      # Embeds chunks and ingests into PostgreSQL (pgvector)
├── eval_recall.py                 # Evaluates retrieval accuracy
└── eval_report.html               # Final evaluation output report


Manual Step-by-Step Execution
If you prefer to run each step manually:

Set Up Virtual Environment & Dependencies:

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
Start the Database:

Bash
cd db_setup
docker compose -f postgress_setup.yaml up -d
cd ..
Set your Hugging Face Token:

Bash
# On Windows (Command Prompt):
set HF_TOKEN=your_huggingface_token
# On macOS/Linux:
export HF_TOKEN=your_huggingface_token
Run the Pipeline Scripts:

Bash
python transripts_diarization.py
python transcript_chunking.py
python ingest.py
python eval_recall.py
View Results:
Open eval_report.html in any web browser to review recall scores and evaluation metrics.
