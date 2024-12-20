from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render, redirect
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
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
import re

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
        
        # Check if both passwords match
        if data['pass1'] != data['pass2']:
            messages.error(request, 'Both Passwords are not the same')
            return redirect('signup')

        # Check if username already exists
        if Auth.objects.filter(username=data['username']).exists():
            messages.error(request, 'Username already exists. Please choose a different username.')
            return redirect('signup')

        # Serialize the data and save if valid
        sz = AuthSerializer(data=data)
        if sz.is_valid():
            sz.save()  # Save new user to the database
            messages.success(request, "You have signed up successfully!")
            return redirect('signin')
        else:
            messages.error(request, "There was an error with your signup. Please check the form.")
    
    return render(request, 'signup.html')

@api_view(['GET', 'POST'])
def signout(request):
    if 'username' in request.session:
        del request.session['username']
        messages.info(request, 'Logged out Successfully')
    return redirect('index')

def index(request):
    return render(request, 'index.html')



# Manually load numpy and joblib
import numpy as np
import joblib

# Function to load models with memory map
def load_model_with_mmap(model_path):
    # Use memory mapping to load the model (this is especially useful for large models)
    return joblib.load(model_path, mmap_mode='r')

# Load the saved model, vectorizer, and label encoder using memory map
model = load_model_with_mmap('trained_model.pkl')
vectorizer = load_model_with_mmap('trained_vectorizer.pkl')
label_encoder = load_model_with_mmap('label_encoder.pkl')

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
from django.contrib import messages
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.shortcuts import render, redirect

def predict_url(request):
    form = URLForm(request.POST or None)
    
    if request.method == 'POST':
        if form.is_valid():
            url = form.cleaned_data['url']

            # Use Django's built-in URLValidator
            validator = URLValidator()

            try:
                validator(url)  # Perform URL validation
            except ValidationError as e:
                # If URL is invalid, add error message using Django's messages framework
                messages.error(request, "Invalid URL format. Please enter a valid URL starting with http:// or https://.")
                return redirect('predict')  # Redirect back to the same page (assumes 'predict' is the name of the view)
            
            feature_vector = analyze_url(url)
            feature_vector = np.array(feature_vector).reshape(1, -1)
            probabilities = model.predict_proba(feature_vector)[0]
            prediction = np.argmax(probabilities)
            predicted_label = label_encoder.inverse_transform([prediction])[0]

            # Store prediction history with the logged-in user
            user = Auth.objects.get(username=request.session.get('username'))
            PredictionHistory.objects.create(user=user, url=url, predicted_label=predicted_label)

            # Generate chart and render as before
            labels = label_encoder.classes_
            fig, ax = plt.subplots()
            ax.bar(labels, probabilities)
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities')
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            graph = base64.b64encode(image_png).decode('utf-8')

            return render(request, 'predict.html', {
                'form': form,
                'url': url,
                'predicted_label': predicted_label,
                'graph': graph
            })

        else:
            # If form is invalid, show a general error message
            messages.error(request, "Invalid form submission. Please check the entered URL.")
            return render(request, 'predict.html', {'form': form})

    else:
        # Handle GET request (initial page load)
        return render(request, 'predict.html', {'form': form})



# @login_required
from django.shortcuts import render
from django.core.exceptions import ObjectDoesNotExist

def history_view(request):
    history = None
    try:
        if 'username' in request.session:
            username = request.session.get('username')
            # print(f"Session username: {username}")  # Debugging output
            user = Auth.objects.get(username=username)
            history = PredictionHistory.objects.filter(user=user).order_by('-prediction_date')
    except ObjectDoesNotExist:
        print("User or prediction history not found.")  # Debugging output
    except Exception as e:
        print(f"Unexpected error: {e}")  # Log any other unexpected errors

    return render(request, 'history.html', {'history': history})


