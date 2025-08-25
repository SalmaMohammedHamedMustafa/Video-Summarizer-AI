# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies: tesseract + ffmpeg for video/audio handling
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app


# Copy requirements file first
COPY requirements.txt .

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment vars
ENV GOOGLE_API_KEY="AIzaSyDdHXpkoF_v8lXFKwz1ce3DeuuIVkW6q9I"


#copy the rest 
COPY main.py my_utils/ audio_processor/ video_text_processor/ summary_creator/ vector_store_builder/ question_answerer/ videos/ ./


# Set the entrypoint to run your program
CMD ["python", "main.py"]