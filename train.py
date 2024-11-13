# Import necessary libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import tempfile
import time

# Define function to analyze content from URLs
def analyze_url(url):
    features = {
        'text': '',
        'image_size': (0, 0),
        'audio_detected': 0,
        'video_detected': 0,
        'audio_duration': 0,
        'video_duration': 0,
        'speech_text': '',
    }

    try:
        response = requests.get(url, timeout=10)
        content_type = response.headers.get('Content-Type', '')

        # Check for text content
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.content, 'html.parser')
            features['text'] = soup.get_text()[:1000]  # Extract first 1000 characters of text
        
        # Check for image content
        if 'image' in content_type:
            try:
                image = Image.open(BytesIO(response.content))
                features['image_size'] = image.size
            except Exception as e:
                print(f"Error processing image: {e}")
        
        # Check for audio content
        if 'audio' in content_type:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio:
                    tmp_audio.write(response.content)
                    tmp_audio.flush()
                    audio = AudioSegment.from_file(tmp_audio.name)
                    features['audio_detected'] = 1
                    features['audio_duration'] = len(audio) / 1000.0  # Duration in seconds
            except Exception as e:
                print(f"Error processing audio: {e}")
        
        # Check for video content
        if 'video' in content_type:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                    tmp_video.write(response.content)
                    tmp_video.flush()
                    video = VideoFileClip(tmp_video.name)
                    features['video_detected'] = 1
                    features['video_duration'] = video.duration  # Duration in seconds
            except Exception as e:
                print(f"Error processing video: {e}")
        
        # Check for speech content
        if 'audio' in content_type:
            try:
                recognizer = sr.Recognizer()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                    tmp_audio.write(response.content)
                    tmp_audio.flush()
                    with sr.AudioFile(tmp_audio.name) as source:
                        audio = recognizer.record(source)
                        try:
                            text = recognizer.recognize_google(audio)
                            features['speech_text'] = text
                        except sr.UnknownValueError:
                            features['speech_text'] = ''
            except Exception as e:
                print(f"Error processing speech recognition: {e}")

    except Exception as e:
        features['text'] = str(e)
    
    return features

# Step 1: Load the dataset
file_path = "MaliciousURL/mali.csv"
dataset = pd.read_csv(file_path)

# Step 2: Apply the content analysis function to the dataset
dataset['features'] = dataset['url'].apply(analyze_url)

# Extracting individual feature columns
text_data = dataset['features'].apply(lambda x: x['text'])
image_sizes = dataset['features'].apply(lambda x: x['image_size'])
audio_detected = dataset['features'].apply(lambda x: x['audio_detected'])
video_detected = dataset['features'].apply(lambda x: x['video_detected'])
audio_duration = dataset['features'].apply(lambda x: x['audio_duration'])
video_duration = dataset['features'].apply(lambda x: x['video_duration'])
speech_text = dataset['features'].apply(lambda x: x['speech_text'])

# Step 3: Convert text content into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
text_features = vectorizer.fit_transform(text_data.fillna('')).toarray()

# Convert image size into numerical features (width, height)
image_features = np.array([list(size) for size in image_sizes])

# Combine all features into a single feature matrix
X = np.hstack((
    text_features,
    image_features, 
    np.array(audio_detected).reshape(-1, 1), 
    np.array(video_detected).reshape(-1, 1),
    np.array(audio_duration).reshape(-1, 1),
    np.array(video_duration).reshape(-1, 1),
    vectorizer.transform(speech_text.fillna('')).toarray()  # Add TF-IDF features for speech text
))

# Encode the 'type' column into numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['type'])

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForest with fewer estimators and parallelization
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

# Measure start time
start_time = time.time()

# Train the model
model.fit(X_train, y_train)

# Measure end time
end_time = time.time()

# Calculate time taken
time_taken = end_time - start_time

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%, Time Taken: {time_taken:.2f} seconds")

# Save the final trained model
model_path = 'MaliciousURL/trained_model.pkl'
joblib.dump(model, model_path)
print(f"Final Model saved at {model_path}")

# Save the vectorizer and label encoder for future use
joblib.dump(vectorizer, 'MaliciousURL/trained_vectorizer.pkl')
joblib.dump(label_encoder, 'MaliciousURL/label_encoder.pkl')
print("Vectorizer and label encoder saved.")
