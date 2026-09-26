# Audio RAG & Diarization Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) pipeline for two-speaker audio conversations. The pipeline transcribes and diarizes audio, chunks dialogue into speaker-aware conversational windows, indexes those chunks for hybrid (keyword + semantic) search, and evaluates retrieval quality against a labeled query set.

## Overview

- **Transcription:** Whisper generates word-level timestamped transcripts.
- **Diarization:** PyAnnote assigns speaker labels (`SPEAKER_00`, `SPEAKER_01`, ...) to each segment. Labels are generic speaker identifiers rather than real names — mapping to actual speaker identity would require a separate voice-identification step against reference samples, which is out of scope for this pipeline.
- **Chunking:** Diarized transcripts are segmented into conversational windows with overlap, preserving cross-speaker context and each chunk's speaker attribution.
- **Indexing:** Chunks are embedded locally via Ollama (`bge-m3`) and stored in PostgreSQL with the `pgvector` extension, alongside a full-text (BM25) index.
- **Retrieval:** Queries run against both the BM25 and vector indexes; results are merged using Reciprocal Rank Fusion (RRF), returning the source file, timestamp range, and speaker for each match.
- **Evaluation:** `eval_recall.py` scores the pipeline against a labeled query set (Recall@5, MRR, nDCG@5, and negative-control abstention), producing `eval_report.html`.

## Project Structure

```
├── data/
│   ├── audio/                     # Raw input audio files (.mp3, .wav)
│   └── transcript_diarized/       # Generated diarization JSON transcripts
│       └── chunked/               # Chunked transcript JSON files
├── db_setup/
│   └── postgress_setup.yaml       # Docker Compose setup for PostgreSQL + pgvector
├── run_pipeline.py                # Cross-platform master execution script
├── requirements.txt                # Python package dependencies
├── transripts_diarization.py      # Transcribes & diarizes audio (Whisper + PyAnnote)
├── transcript_chunking.py         # Chunks diarized transcripts for search indexing
├── ingest.py                      # Embeds chunks and ingests into PostgreSQL (pgvector)
├── eval_recall.py                 # Evaluates retrieval accuracy
└── eval_report.html               # Final evaluation output report
```

## Setup & Usage

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the database

```bash
cd db_setup
docker compose -f postgress_setup.yaml up -d
cd ..
```

### 3. Set your Hugging Face token

PyAnnote's diarization model is gated on Hugging Face; a token is required to download it.

```bash
# Windows (Command Prompt)
set HF_TOKEN=your_huggingface_token

# macOS/Linux
export HF_TOKEN=your_huggingface_token
```

### 4. Run the pipeline

```bash
python transripts_diarization.py
python transcript_chunking.py
python ingest.py
python eval_recall.py
```

> **Note:** `transripts_diarization.py` can be skipped if diarized transcripts already exist under `data/transcript_diarized/` (e.g. from a previous run or the files included in this repo). In that case, start directly from `transcript_chunking.py`.

Or run all steps in one go:

```bash
python run_pipeline.py
```

### 5. View results

Open `eval_report.html` in a browser to review recall, MRR, nDCG, and negative-control results.

## Notes & Limitations

- Diarization accuracy degrades on overlapping speech, occasionally causing speaker-boundary mixups at turn transitions; a post-processing step merges speaker-turn artifacts shorter than 0.8s to reduce this.
- Local embedding throughput depends on host GPU/VRAM availability.
- ASR homophone/acoustic misspellings can reduce exact-match BM25 accuracy for affected terms.
- Speaker labels are generic (`SPEAKER_00`/`SPEAKER_01`) rather than real names; see Overview above.
