import cv2
import pytesseract



from moviepy import VideoFileClip

def extract_frames_from_video(video_path, interval=3):
    frames = []
    clip = VideoFileClip(video_path)
    duration = clip.duration  

    for t in range(0, int(duration), interval):
        frame = clip.get_frame(t)
        frames.append(frame)

    return frames



def ocr_extract_text_from_frames(frames):
    """
    Extracts text from a list of frames using OCR.

    Parameters:
    frames (list): List of frames as images.

    Returns:
    list: List of extracted text from each frame.
    """
    texts = []
    for frame in frames:
        text = pytesseract.image_to_string(frame)
        texts.append(text)
    return texts

def remove_repeated_text(texts):
    seen = set()
    unique_texts = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    return unique_texts
