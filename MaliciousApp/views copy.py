from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render, redirect
from rest_framework import status
from .models import *
from .serializers import *
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import HttpResponse
import joblib
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image
from .forms import URLForm
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import tempfile
from .models import *
import matplotlib.pyplot as plt
import base64
from django.contrib.auth import authenticate, login


# Create your views here.
def home(request):
    return render(request, 'home.html')

@api_view(['GET', 'POST'])
def signin(request):
    uname = ''
    if request.method == 'POST':
        data = {
            "username": request.POST.get("username"),
            "password": request.POST.get("password"),
        }
        uname = data['username']
        try:
            user = Auth.objects.get(username=data['username'], pass1=data['password'])
            request.session['username'] = user.username
            messages.success(request, "Logged Successfully")
            return redirect('index')
        except Auth.DoesNotExist:
            messages.error(request, "Invalid Credentials")
            return redirect('signin')
    context = {'username': uname}
    return render(request, 'signin.html', context)

@api_view(['GET', 'POST'])
def signup(request):
    if request.method == 'POST':
        data = {
            "username": request.POST.get("username"),
            "email": request.POST.get("email"),
            "pass1": request.POST.get("pass1"),
            "pass2": request.POST.get("pass2")
        }
        if data['pass1'] != data['pass2']:
            messages.error(request, 'Both Passwords are not same')
            return redirect('signup')
        sz = AuthSerializer(data=data)
        if sz.is_valid():
            sz.save()
            messages.success(request, "You have signed up successfully!")
            return redirect('signin')
        else:
            messages.error(request, "There was an error with your signup.")
    return render(request, 'signup.html')

@api_view(['GET', 'POST'])
def signout(request):
    if 'username' in request.session:
        del request.session['username']
        messages.info(request, 'Logged out Successfully')
    return redirect('index')

def index(request):
    return render(request, 'index.html')

# Load the saved model, vectorizer, and label encoder
model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('trained_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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

        # Check for speech content in audio
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

    # Convert features to numerical format
    text_features = vectorizer.transform([features['text']]).toarray()
    image_features = np.array([list(features['image_size'])])
    audio_detected = np.array([[features['audio_detected']]])
    video_detected = np.array([[features['video_detected']]])
    audio_duration = np.array([[features['audio_duration']]])
    video_duration = np.array([[features['video_duration']]])
    speech_features = vectorizer.transform([features['speech_text']]).toarray()

    # Combine all features into a single feature vector
    feature_vector = np.hstack((
        text_features,
        image_features,
        audio_detected,
        video_detected,
        audio_duration,
        video_duration,
        speech_features
    ))

    return feature_vector

# @login_required  
def predict_url(request):
    if request.method == 'POST':
        form = URLForm(request.POST)
        if form.is_valid():
            url = form.cleaned_data['url']
            feature_vector = analyze_url(url)

            # Ensure feature_vector is a 2D array of shape (1, n_features)
            feature_vector = np.array(feature_vector).reshape(1, -1)

            # Predict the label and probabilities
            probabilities = model.predict_proba(feature_vector)[0]  # Get probabilities
            prediction = np.argmax(probabilities)  # Index of the highest probability
            predicted_label = label_encoder.inverse_transform([prediction])[0]

            # Store prediction history
            PredictionHistory.objects.create(url=url, predicted_label=predicted_label)

            # Generate a bar chart for the probabilities
            labels = label_encoder.classes_  # Get the list of labels
            fig, ax = plt.subplots()
            ax.bar(labels, probabilities)
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')

            # Save the plot to a PNG image in memory
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Convert the PNG image to base64 string
            graph = base64.b64encode(image_png).decode('utf-8')

            # Render the result within the same template
            return render(request, 'predict.html', {
                'form': form,
                'url': url,
                'predicted_label': predicted_label,
                'graph': graph  # Pass the graph to the template
            })
    else:
        form = URLForm()

    return render(request, 'predict.html', {'form': form})


# @login_required
def history_view(request):
    history = PredictionHistory.objects.all().order_by('-prediction_date')
    return render(request, 'history.html', {'history': history})
