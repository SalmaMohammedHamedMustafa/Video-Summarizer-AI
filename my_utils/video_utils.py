from moviepy import VideoFileClip

def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from a video file and saves it as an MP3 file.

    Parameters:
    video_path (str): Path to the input video file.
    audio_path (str): Path where the extracted audio will be saved.
    """
    with VideoFileClip(video_path) as video:
        audio = video.audio
        audio.write_audiofile(audio_path, codec='mp3')
        video.close()
        audio.close()