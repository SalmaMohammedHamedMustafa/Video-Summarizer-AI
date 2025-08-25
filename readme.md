# Video Summarizer AI

This project implements an agentic AI system to extract, transcribe, OCR, and summarize video content. It uses modern generative AI models combined with multi-agent orchestration to produce both concise summaries and detailed documentation from multiple input modalities.

## Features

- **Audio Extraction**
  Extract audio tracks from video files using MoviePy (`video_utils.py`).
- **Audio Transcription**
  Transcribe extracted audio using OpenAI Whisper with support for chunked transcription of long files (`audio_processor.py`).
- **Frame Extraction & OCR**
  Extract video frames at regular intervals and perform OCR with Tesseract OCR to capture visible text in the video frames (`video_text_processor.py`).
- **Duplicate OCR Text Removal**
  Automatically remove repeated OCR-extracted texts to reduce redundancy (`video_text_processor.py`).
- **Agentic Summarization with LangGraph & Google Gemini**
  A multi-agent AI workflow that processes and merges information from OCR and transcription sources intelligently (`summary_creator.py`):
  - **OCR Agent**: Handles incomplete or noisy OCR text extraction with uncertainty awareness.
  - **Transcription Agent**: Produces detailed, accurate documentation from audio transcription.
  - **Fusion Agent**: Combines outputs from OCR and transcription agents to produce:
    - A concise summary.
    - A comprehensive full documentation.
- **Vector Store Construction for Semantic Search**
  Builds and saves a FAISS vector store from OCR and transcription texts to enable efficient similarity search and question answering (`vector_store_builder.py`).
- **Interactive Question Answering**
  Enables users to query the content of videos by searching the vector store and answering questions via Google Gemini generative AI (`question_answerer.py`).
- **Chunked Audio Transcription**
  For long audio files, transcription is processed chunk-by-chunk to handle resource constraints and improve performance (`audio_processor.py`).
- **Configurable File and Directory Management**
  Automatically manages directories and file paths for audios, transcriptions, OCR texts, summaries, full docs, and vector stores.
- **Warning and Log Suppression**
  Suppresses unnecessary warnings and logs for cleaner console output.

## Modules Overview

- **`audio_processor.py`**
  Handles loading audio files, chunked transcription with Whisper, and ensuring correct sampling rates.
- **`video_utils.py`**
  Utilities for extracting audio tracks from video files using MoviePy.
- **`video_text_processor.py`**
  Captures frames from video at configurable intervals, applies OCR, and removes duplicate extracted texts.
- **`summary_creator.py`**
  Implements a LangGraph-based multi-agent AI workflow using Google Gemini LLM for summarization and documentation fusion.
- **`vector_store_builder.py`**
  Builds FAISS vector indices from text files using HuggingFace embeddings, facilitating semantic search.
- **`question_answerer.py`**
  Provides a question-answering interface over the built vector store leveraging Google Gemini generative AI.
- **`main.py`**
  Orchestrates the full pipeline — from audio extraction, transcription, frame extraction, OCR, summarization, vector store creation, to interactive Q&A.

## System Architecture

1. **Preprocessing**
   - Extract audio track from the video file.
   - Perform transcription on the extracted audio (supports chunking).
   - Extract frames from the video at intervals and perform OCR.
   - Remove duplicate OCR texts.
2. **Agentic AI Graph**
   - OCR agent extracts reliable insights from possibly noisy OCR text.
   - Transcription agent produces accurate and detailed documentation.
   - Fusion agent intelligently combines OCR summary and transcription documentation into:
     - Concise summary file.
     - Detailed full documentation file.
3. **Vector Store & Q&A**
   - Build a FAISS vector store from OCR and transcription texts.
   - Enable semantic similarity searches and user question answering.

- Here is the full updated **Setup and Installation** section including instructions for both native Python environment use and Docker usage, integrated with your existing README style:

  ------

  ## Setup and Installation

  ## Option 1: Native Python Environment

  ## Prerequisites

  - Python 3.10 or newer installed on your system
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system (not just the Python package)

  On Ubuntu/Debian, install Tesseract OCR using:

  ```
  bash
  sudo apt-get install tesseract-ocr
  ```

  ## Install Python Dependencies

  Install the required Python packages using pip:

  ```
  bash
  pip install -r requirements.txt
  ```

  ## Configuration

  Set your Google Gemini API key as an environment variable:

  ```
  bash
  export GOOGLE_API_KEY="your_gemini_api_key_here"
  ```

  ## Usage

  1. Place your video file in a known directory.
  2. Run the pipeline by executing:

  ```
  bash
  python main.py
  ```

  1. When prompted, enter the path to your video file.

  The system will:

  - Extract audio and save it as MP3.
  - Transcribe audio (supports chunked transcription).
  - Extract frames at intervals and perform OCR.
  - Remove duplicate OCR texts.
  - Run the multi-agent summarization workflow.
  - Save summary and full documentation to the `results/` directory.
  - Build vector store index for semantic search.
  - Support optional interactive question answering on video content.

  ------

  ## Option 2: Using Docker (Recommended for Simplified Setup)

  Your project includes a Dockerfile which builds a container image with all system and Python dependencies pre-installed, including Tesseract OCR and FFmpeg.

  ## Build the Docker Image

  In your project root directory (where the Dockerfile and requirements.txt reside), run:

  ```
  bash
  docker build -t video-summarizer-ai .
  ```

  ## Run the Docker Container

  Run the container interactively, optionally mounting your local videos directory:

  ```
  bash
  docker run -it --rm -v /path/to/local/videos:/app/videos video-summarizer-ai
  ```

  Inside the container, when prompted for the video path, provide the relative path (e.g., `videos/sample_video.mp4`).

  ------

  ## Output Files

  - Audio: `audios/{video_name}.mp3`
  - Transcription: `transcriptions/{video_name}_transcription.txt`
  - OCR Texts: `video_texts/{video_name}_texts.txt`
  - Summary: `results/{video_name}_summary.txt`
  - Full Documentation: `results/{video_name}_full_doc.txt`
  - Vector Store: `vector_stores/{video_name}_vector_store/`