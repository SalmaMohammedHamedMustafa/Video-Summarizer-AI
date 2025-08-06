import cv2
import pytesseract
from moviepy import VideoFileClip

class VideoTextProcessor:
    def __init__(self, video_path, interval=3):
        self.video_path = video_path
        self.interval = interval
        self.frames = []
        self.texts = []

    def extract_frames(self):
        self.frames = []
        clip = VideoFileClip(self.video_path)
        duration = clip.duration
        for t in range(0, int(duration), self.interval):
            frame = clip.get_frame(t)
            self.frames.append(frame)
        return self.frames

    def ocr_extract_text(self):
        if not self.frames:
            self.extract_frames()
        self.texts = []
        for frame in self.frames:
            text = pytesseract.image_to_string(frame)
            self.texts.append(text)
        return self.texts

    def remove_repeated_text(self):
        seen = set()
        unique_texts = []
        for text in self.texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        return unique_texts
