from my_utils import extract_audio_from_video
from audio_processor import transcribe_audio
from video_text_processor import extract_frames_from_video, ocr_extract_text_from_frames, remove_repeated_text
from agentic_system import summarize_video
import os


def print_final_output(final_output):
    for key, value in final_output.items():
        print(f"\n=== {key.upper()} ===\n")
        print(value)
        print("\n" + "="*40)  # separator line for readability

def main():


    video_path = input("Enter the path to the video file: ").strip()

    # === Compute derived paths ===
    base_name = os.path.splitext(os.path.basename(video_path))[0]  # e.g. "AI"
    audio_path = f"audios/{base_name}.mp3"
    transcription_path = f"transcriptions/{base_name}_transcription.txt"
    ocr_text_path = f"video_texts/{base_name}_texts.txt"

    # === Step 1: Extract audio ===
    if os.path.exists(audio_path):
        print("Audio already extracted. Skipping audio extraction.")
    else:
        print("Extracting audio from the video...")
        extract_audio_from_video(video_path, audio_path)

    # === Step 2: Transcribe audio ===
    if os.path.exists(transcription_path):
        print("Transcription already exists. Skipping transcription.")
    else:
        print("Transcribing audio...")
        transcription = transcribe_audio(audio_path, model_name="small", chunk_duration=30)
        os.makedirs("transcriptions", exist_ok=True)
        with open(transcription_path, "w") as f:
            f.write(transcription)

    # === Step 3: Extract and OCR frames ===
    if os.path.exists(ocr_text_path):
        print("OCR already exists. Skipping OCR.")
    else:
        print("Extracting frames and performing OCR...")
        frames = extract_frames_from_video(video_path, interval=3)
        texts = ocr_extract_text_from_frames(frames)
        unique_texts = remove_repeated_text(texts)
        os.makedirs("video_texts", exist_ok=True)
        with open(ocr_text_path, "w") as f:
            f.writelines(unique_texts)

    # === Step 4: Summarize ===
    print("Summarizing the video...")
    final_output = summarize_video(
        ocr_path=ocr_text_path,
        transcript_path=transcription_path
    )
    print_final_output(final_output)



if __name__ == "__main__":
    main()