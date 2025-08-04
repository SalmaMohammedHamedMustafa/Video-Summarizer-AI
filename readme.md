```markdown
# Video Summarizer AI

This project implements an agentic AI system to extract, transcribe, OCR, and summarize video content. It uses modern generative AI models combined with multi-agent orchestration to produce both concise summaries and detailed documentation from multiple input modalities.

---

## Features

- **Audio Extraction**: Extracts the audio track from video files using MoviePy.
- **Audio Transcription**: Transcribes extracted audio with OpenAI Whisper, supporting chunked transcription for long files.
- **Frame Extraction & OCR**: Extracts frames from videos and applies OCR with Tesseract to capture visible text in the video frames.
- **Agentic Summarization**: Uses Google Gemini generative AI models orchestrated via LangGraph to:
  - Process OCR text and transcriptions as separate agents.
  - Fuse outputs intelligently with contextual awareness of OCR uncertainty.
  - Generate two output formats:
    - Concise summary.
    - Full, detailed documentation combining all sources.
- **Customizable Workflow**: Easily extendable LangGraph workflow with clearly defined node agents.

---

## System Architecture

1. **Preprocessing**
    - Extract audio from the video.
    - Transcribe audio using Whisper.
    - Extract video frames.
    - Perform OCR on extracted frames.
    - Remove duplicate OCR texts.

2. **Agentic AI Graph**
    - **OCR Agent**: Summarizes noisy/incomplete OCR with uncertainty awareness.
    - **Transcription Agent**: Creates accurate full documentation of audio text.
    - **Fusion Agent**: Merges both outputs to generate summary and full doc.

3. **Output**
    - Writes concise summary and full documentation to separate files.
    - Prints output locations.

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system (not a Python package).

On Ubuntu/Debian:
```
sudo apt-get install tesseract-ocr
```

### Install Python Dependencies

```
pip install -r requirements.txt
```

Example `requirements.txt` includes:

```
langchain-google-genai==0.0.1
langgraph==0.0.63
google-generativeai==1.5.0
whisper==1.0
librosa==0.10.0
numpy==1.26.4
tqdm==4.67.1
moviepy==2.2.1
opencv-python==4.7.0.72
pytesseract==0.3.10
```

---

## Configuration

### Google Gemini API

Set your Gemini API key as environment variable:

```
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

Or define it in your script before running the agentic system.

---

## Usage

### Run the Pipeline

1. Place your video in a known directory.

2. Run the `main.py` script:

```
python main.py
```

3. Enter the path to your video file when prompted.

The pipeline will:

- Extract audio, transcribe audio.
- Extract frames and OCR text.
- Run agentic summarization.
- Save outputs to `results/summary.md` and `results/full_doc.md`.

---

## Code Overview

- `main.py`: Orchestrates your workflow from video to summary.
- `audio_processor.py`: Whisper-based transcription with chunking.
- `video_utils.py`: Video audio extraction utilities.
- `video_text_processor.py`: Frame extraction and OCR processing.
- `agentic_system.py`: LangGraph-based multi-agent workflow using Google Gemini LLM for summarization.

