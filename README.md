# 🎙️ Voice Classification

A Python-based audio (voice) classification system that uses a trained machine learning model to classify voice/audio inputs.

---

## 🔍 Overview

This project loads a pre-trained audio classifier and provides an interface (via `app.py`) to classify voice/audio samples. It includes exploratory/training work in `code.ipynb` and a serialized model file `audio_classifier_model.pkl` used for inference.

---

## 🧰 Key Components

| File / Folder                | Purpose                                                                                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `app.py`                     | The main application script. Likely handles loading the model, processing input audio files, extracting features, and displaying classification results.           |
| `audio_classifier_model.pkl` | The serialized (pickled) model used for classifying audio samples. Contains both the model weights & the pipeline needed to preprocess input and make predictions. |
| `code.ipynb`                 | Notebook for experimentation, data preprocessing, feature extraction, model training / evaluation. Serves as development backup / documentation.                   |

---

## ⚙️ Workflow (Assumed)

Here’s a likely workflow of how the system works, from raw audio input → classification output:

1. **Audio Input**

   * User provides an audio sample (e.g. `.wav`, `.mp3`), either via file upload or through a specific directory.

2. **Preprocessing / Feature Extraction**

   * Audio is converted / cleaned (e.g. ensuring correct sampling rate, trimming silence, normalizing amplitude).
   * Features are extracted. Could be MFCCs (Mel Frequency Cepstral Coefficients), spectral features, chroma features, or other relevant audio descriptors.

3. **Loading the Model**

   * The pickled model (`audio_classifier_model.pkl`) is loaded.
   * The model likely includes both preprocessing steps + the classifier (so whatever features training used and the machine learning algorithm are encapsulated).

4. **Inference**

   * The extracted features are fed to the model.
   * The model produces a classification label (e.g. speaker identity, type of voice, or another category depending on what the classes are).
   * Optionally, probabilities or confidence scores may be given.

5. **Output / Interface**

   * `app.py` provides interface (CLI, GUI or Web) for users to supply audio, run classification, and see the results.
   * Possibly shows some metadata: duration of audio, file name, maybe feature values or prediction confidence.

6. **(Optional) Model Training / Evaluation**

   * The notebook (`code.ipynb`) provides exploratory steps: loading datasets, splitting into training / test, feature extraction, training the model, evaluating performance (accuracy, confusion matrix, etc.).

---

## 🛠️ Tech Stack & Dependencies

* Language: Python
* Serialization: Pickle for model (`.pkl`)
* Libraries likely used (based on typical audio classification projects):

  * `librosa` or similar for audio processing / feature extraction
  * `scikit-learn` for model training / inference
  * Possibly `numpy`, `pandas` for data handling
  * `joblib` or `pickle` for saving & loading models
* Interface: Possibly simple command-line / script (app.py)

---

## 🎯 Use Cases

* Classifying voice recordings (e.g. speaker identification)
* Detecting types of voice (e.g. male/female, emotion, accent etc.) depending on what classes trained
* Could be used in applications needing audio classification (voice assistants, audio surveillance, voice recognition)

---

## 📂 Project Structure (Suggested)

```
voice-classification/
├── app.py
├── audio_classifier_model.pkl
├── code.ipynb
├── /data/        (if there are audio samples / datasets used)
├── requirements.txt
└── README.md
```

---

## ✅ How to Run

Here are suggested steps to use / test this project:

```bash
# Clone the repo
git clone https://github.com/dilanmalaviarachchi/voice-clasification.git
cd voice-clasification

# Install dependencies
pip install -r requirements.txt

# Run the app (this could be CLI or Web, depending on app.py)
python app.py --input path/to/audiofile.wav
# Or possibly
streamlit run app.py
```

Replace `path/to/audiofile.wav` with your audio file.

---

## 📈 Future Enhancements (Ideas)

* Add support for multiple input formats (wav, mp3, live audio / microphone)
* Show confidence / probability scores in the output
* Visualize audio features (spectrograms, MFCC plots) in the app or notebook
* Allow retraining / fine-tuning the model with new data via the notebook
* Deploy as a web service (Flask, FastAPI, or via web-UI)

---


