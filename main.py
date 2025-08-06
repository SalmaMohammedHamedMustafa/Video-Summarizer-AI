from my_utils import extract_audio_from_video
from audio_processor import AudioProcessor
from video_text_processor import VideoTextProcessor
from summary_creator import VideoSummarizer
from vector_store_builder import VectorStoreBuilder
from question_answerer import QuestionAnswerer
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
    summary_path = f"results/{base_name}_summary.txt"
    full_doc_path = f"results/{base_name}_full_doc.txt"
    vector_store_path = f"vector_stores/{base_name}_vector_store"

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
        audio_processor = AudioProcessor(model_name="small")
        print("Transcribing audio...")
        transcription = audio_processor.transcribe(audio_path, chunk_duration=30)
        os.makedirs("transcriptions", exist_ok=True)
        with open(transcription_path, "w") as f:
            f.write(transcription)

    # === Step 3: Extract and OCR frames ===
    if os.path.exists(ocr_text_path):
        print("OCR already exists. Skipping OCR.")
    else:
        video_text_processor = VideoTextProcessor(video_path, interval=3)
        print("Extracting frames and performing OCR...")
        frames = video_text_processor.extract_frames()
        texts = video_text_processor.ocr_extract_text()
        unique_texts = video_text_processor.remove_repeated_text()
        os.makedirs("video_texts", exist_ok=True)
        with open(ocr_text_path, "w") as f:
            f.writelines(unique_texts)

    # === Step 4: Summarize ===
    if os.path.exists(summary_path) and os.path.exists(full_doc_path):
        print("Summary and full documentation already exist. Skipping summarization.")
    else:
        video_summarizer = VideoSummarizer(
            ocr_path=ocr_text_path,
            transcript_path=transcription_path,
            summary_out=summary_path,
            full_doc_out=full_doc_path
        )
        print("Summarizing the video...")
        final_output = video_summarizer.summarize()


    # === Step 5: Build vector store ===
    if os.path.exists(vector_store_path):
        print("Vector store already exists. Skipping vector store creation.")
    else:
        vector_store_builder = VectorStoreBuilder()
        print("Building vector store...")
        doc_paths = [ocr_text_path, transcription_path]
        vector_store_builder.build_and_save(doc_paths, vector_store_path)
        print(f"Vector store saved at: {vector_store_path}")

    # === Step 6: Optional Question Answering ===
    user_input = input("Do you want to ask a question about the video? (yes/no): ").strip().lower()
    if user_input == "yes":
        question_answerer = QuestionAnswerer(vector_store_path)
        while True:
            question = input("Enter your question (or type 'quit' to exit): ").strip()
            if question.lower() == "quit":
                break
            answer = question_answerer.ask(question)
            print("\n=== ANSWER ===\n")
            print(answer)




if __name__ == "__main__":
    main()