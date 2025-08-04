import whisper
import librosa
import numpy as np
import warnings
from tqdm import tqdm
import os

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

def transcribe_audio(audio_path, model_name="small", chunk_duration=30):
    """
    Advanced transcription with better chunking using librosa and progress tracking.
    
    Parameters:
    audio_path (str): Path to the audio file to be transcribed.
    model_name (str): Name of the Whisper model to use.
    chunk_duration (int): Duration of each chunk in seconds (default is 30).
    
    Returns:
    str: Transcribed text from the audio.
    """
    try:
        print(f"Loading Whisper model '{model_name}'...")
        # Load the Whisper model with fp16=False to avoid CPU warning
        model = whisper.load_model(model_name)
        
        print(f"Loading audio file: {audio_path}")
        # Load audio with librosa to get duration
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        print(f"Audio duration: {duration:.2f} seconds")
        
        # If audio is longer than chunk_duration, process in chunks
        if duration > chunk_duration:
            print(f"Audio is longer than {chunk_duration}s, processing in chunks...")
            
            transcriptions = []
            samples_per_chunk = int(chunk_duration * sr)
            total_chunks = int(np.ceil(len(y) / samples_per_chunk))
            
            print(f"Total chunks to process: {total_chunks}")
            
            # Create progress bar
            with tqdm(total=total_chunks, desc="Transcribing chunks", unit="chunk") as pbar:
                for i in range(0, len(y), samples_per_chunk):
                    chunk_num = (i // samples_per_chunk) + 1
                    chunk = y[i:i + samples_per_chunk]
                    
                    # Update progress bar description
                    pbar.set_description(f"Processing chunk {chunk_num}/{total_chunks}")
                    
                    # Whisper expects 16kHz audio, so resample if necessary
                    if sr != 16000:
                        chunk_resampled = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
                    else:
                        chunk_resampled = chunk
                    
                    # Transcribe the chunk with explicit parameters to avoid warnings
                    result = model.transcribe(
                        chunk_resampled,
                        language="en",  # Specify English
                        fp16=False,     # Avoid FP16 warning on CPU
                        verbose=False   # Suppress Whisper's own progress output
                    )
                    
                    if result['text'].strip():  # Only add non-empty transcriptions
                        transcriptions.append(result['text'].strip())
                    
                    # Update progress bar
                    pbar.update(1)
            
            print("✓ Chunked transcription completed!")
            return " ".join(transcriptions)
        else:
            print("Processing single audio file...")
            # For shorter audio, transcribe directly
            with tqdm(total=1, desc="Transcribing audio", unit="file") as pbar:
                result = model.transcribe(
                    audio_path,
                    language="en",  # Specify English
                    fp16=False,     # Avoid FP16 warning on CPU
                    verbose=False   # Suppress Whisper's own progress output
                )
                pbar.update(1)
            
            print("✓ Transcription completed!")
            return result['text'].strip()
            
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return ""