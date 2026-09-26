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
