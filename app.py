import streamlit as st
import numpy as np
import librosa
import pickle
import soundfile as sf
import os

# Page config
st.set_page_config(page_title="🎧 Voice Classifier", layout="centered")

# Custom styles
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: #ffffff;
        background: linear-gradient(to right, #6a11cb, #2575fc);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .result-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        color: #333333;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-in;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🎧 Voice Emotion/Class Predictor</div>', unsafe_allow_html=True)

# Load model silently
try:
    with open("audio_classifier_model.pkl", "rb") as f:
        model = pickle.load(f)
    model_loaded = True
except:
    model_loaded = False

# Feature extraction
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    try:
        with sf.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            result = np.array([])
            if chroma or mel:
                stft = np.abs(librosa.stft(X))
            if mfcc:
                mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
                if mfccs.shape[1] > 0:
                    result = np.hstack((result, np.mean(mfccs.T, axis=0)))
            if chroma:
                chroma_feat = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
                if chroma_feat.shape[1] > 0:
                    result = np.hstack((result, np.mean(chroma_feat.T, axis=0)))
            if mel:
                mel_feat = librosa.feature.melspectrogram(y=X, sr=sample_rate)
                if mel_feat.shape[1] > 0:
                    result = np.hstack((result, np.mean(mel_feat.T, axis=0)))
            return result
    except:
        return None

# Upload section
st.subheader("📤 Upload Your Voice (.wav)")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file and model_loaded:
    st.audio(uploaded_file, format="audio/wav")

    # Save temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Feature extraction
    features = extract_feature("temp.wav")

    # Check if extraction was successful
    if features is not None:
        features = features.reshape(1, -1)
        if features.shape[1] == model.n_features_in_:
            prediction = model.predict(features)
            st.markdown(f'<div class="result-box">🎯 **Predicted Class:** {prediction[0]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box">⚠️ Invalid audio input. Please try another file.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box">⚠️ Couldn\'t process this audio file.</div>', unsafe_allow_html=True)

    # Clean temp file
    try:
        os.remove("temp.wav")
    except:
        pass

elif not model_loaded:
    st.markdown('<div class="result-box">❌ Model not found. Please add <code>audio_classifier_model.pkl</code>.</div>', unsafe_allow_html=True)
