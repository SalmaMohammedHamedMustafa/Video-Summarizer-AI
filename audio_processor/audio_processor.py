import whisper
import librosa
import numpy as np
import warnings
from tqdm import tqdm
import os

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

class AudioProcessor:
    def __init__(self, model_name="small"):
        """
        Initialize the AudioProcessor with a specified Whisper model.
        """
        print(f"Loading Whisper model '{model_name}'...")
        self.model = whisper.load_model(model_name)
        self.model_name = model_name

    def transcribe(self, audio_path, chunk_duration=30):
        """
        Transcribe the given audio file, processing in chunks if necessary.

        Parameters:
        audio_path (str): Path to the audio file to be transcribed.
        chunk_duration (int): Duration of each chunk in seconds (default is 30).

        Returns:
        str: Transcribed text from the audio.
        """
        try:
            print(f"Loading audio file: {audio_path}")
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            print(f"Audio duration: {duration:.2f} seconds")

            if duration > chunk_duration:
                print(f"Audio is longer than {chunk_duration}s, processing in chunks...")
                transcriptions = []
                samples_per_chunk = int(chunk_duration * sr)
                total_chunks = int(np.ceil(len(y) / samples_per_chunk))
                print(f"Total chunks to process: {total_chunks}")

                with tqdm(total=total_chunks, desc="Transcribing chunks", unit="chunk") as pbar:
                    for i in range(0, len(y), samples_per_chunk):
                        chunk_num = (i // samples_per_chunk) + 1
                        chunk = y[i:i + samples_per_chunk]
                        pbar.set_description(f"Processing chunk {chunk_num}/{total_chunks}")

                        if sr != 16000:
                            chunk_resampled = librosa.resample(chunk, orig_sr=sr, target_sr=16000)
                        else:
                            chunk_resampled = chunk

                        result = self.model.transcribe(
                            chunk_resampled,
                            language="en",
                            fp16=False,
                            verbose=False
                        )

                        if result['text'].strip():
                            transcriptions.append(result['text'].strip())
                        pbar.update(1)

                print("✓ Chunked transcription completed!")
                return " ".join(transcriptions)
            else:
                print("Processing single audio file...")
                with tqdm(total=1, desc="Transcribing audio", unit="file") as pbar:
                    result = self.model.transcribe(
                        audio_path,
                        language="en",
                        fp16=False,
                        verbose=False
                    )
                    pbar.update(1)
                print("✓ Transcription completed!")
                return result['text'].strip()

        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return ""
