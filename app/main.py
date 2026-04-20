from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import whisper
import uuid
from gtts import gTTS
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from datetime import datetime
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


app = FastAPI(title="Voice AI API")


# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# 📁 FOLDERS
# -------------------------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------
# 🎤 LOAD WHISPER MODEL (ONCE)
# -------------------------------
model = whisper.load_model("base")

# -------------------------------
# 🧠 LOAD EMOTION MODEL
# -------------------------------
emotion_model = tf.keras.models.load_model("app/model/cnn_bilstm_2.keras")

# Emotion Labels (CHANGE if your model uses different order)
EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

# -------------------------------
# 🧪 TEST
# -------------------------------
@app.get("/")
def root():
    return {"message": "Backend is working!"}

# -------------------------------
# 🧪 Converting audio into melspectrogram + MFFC features
# -------------------------------

def extract_features(file_path, max_len=200):
    y, sr = librosa.load(file_path, sr=16000)

    # Mel (128)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel)

    # MFCC (13)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    def pad(x):
        if x.shape[1] < max_len:
            return np.pad(x, ((0,0),(0,max_len-x.shape[1])))
        return x[:, :max_len]

    mel_db = pad(mel_db)
    mfcc = pad(mfcc)

    combined = np.vstack((mel_db, mfcc))  # (141, 200)

    # SAME normalization
    combined = (combined - np.mean(combined)) / (np.std(combined) + 1e-8)

    return combined.reshape(1, 141, 200, 1)


# =====================================================
# 🎭 EMOTION DETECTION (FINAL FEATURE)
# =====================================================
@app.post("/emotion-detection")
async def emotion_detection(file: UploadFile = File(...)):
    try:
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        input_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ CORRECT FEATURE EXTRACTION
        features = extract_features(input_path)

        # ✅ PREDICTION
        predictions = emotion_model.predict(features)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        emotion = EMOTION_LABELS[predicted_index]

        # cleanup
        os.remove(input_path)

        return {
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "all_scores": predictions.tolist()
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# =====================================================
# 🎤 VOICE → TEXT (WHISPER)
# =====================================================
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):


    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported format")


    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)


    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    try:
        result = model.transcribe(file_path)
        return {"transcript": result["text"]}


    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# =====================================================
# 🔊 TEXT → VOICE (gTTS)
# =====================================================
@app.post("/text-to-voice")
async def text_to_voice(
    text: str = Form(...),
    voice_id: str = Form(...),
    speed: float = Form(...),
    pitch: int = Form(...)
):
    try:
        file_name = f"{uuid.uuid4()}.mp3"
        file_path = os.path.join(OUTPUT_DIR, file_name)


        tts = gTTS(text=text, lang='en', slow=(speed < 1.0))
        tts.save(file_path)


        return {
            "audio_url": f"http://127.0.0.1:8000/download/{file_name}"
        }


    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# -------------------------------
# 🎧 SERVE AUDIO
# -------------------------------
@app.get("/audio/{file_name}")
def get_audio(file_name: str):
    file_path = os.path.join(OUTPUT_DIR, file_name)


    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)


    return FileResponse(
    file_path,
    media_type="audio/mpeg",
    headers={"Content-Disposition": "inline"}
    )






# =====================================================
# 🎤 VOICE Enhancement
# =====================================================




@app.post("/enhance-audio")
async def enhance_audio(
    file: UploadFile = File(...),
    clarityLevel: str = Form(...),
    normalizeVolume: bool = Form(...)
):
    try:
        # ✅ Unique file name
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        input_path = os.path.join(UPLOAD_DIR, unique_name)


        # Save file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)


        # Load audio
        y, sr = librosa.load(input_path, sr=None)


        # 🎚️ CLARITY ENHANCEMENT
        if clarityLevel == "Low":
            y = librosa.effects.preemphasis(y, coef=0.8)
        elif clarityLevel == "Medium":
            y = librosa.effects.preemphasis(y, coef=0.9)
        elif clarityLevel == "High":
            y = librosa.effects.preemphasis(y, coef=0.97)


        # ✅ Volume Boost (NEW 🔥)
        y = y * 3   # increase loudness


        # 🔊 VOLUME NORMALIZATION
        if normalizeVolume:
            y = y / np.max(np.abs(y))


        # Save enhanced file
        output_filename = f"enhanced_{unique_name}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)


        from pydub import AudioSegment


        # Save temporary WAV
        temp_wav = os.path.join(OUTPUT_DIR, f"temp_{unique_name}.wav")
        sf.write(temp_wav, y, sr)


        # Convert WAV → MP3
        output_filename = f"enhanced_{unique_name}.mp3"
        output_path = os.path.join(OUTPUT_DIR, output_filename)


        audio = AudioSegment.from_wav(temp_wav)
        audio.export(output_path, format="mp3", bitrate="192k")


        # Remove temp file
        os.remove(temp_wav)


        return {
    "enhanced_url": f"http://127.0.0.1:8000/download/{output_filename}"
}


    except Exception as e:
        return {"error": str(e)}




@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(file_path, media_type='audio/mpeg', filename=filename)








# =====================================================
# 🔇 ADVANCED NOISE REDUCTION (NEW FEATURE)
# =====================================================


@app.post("/noise-reduction")
async def noise_reduction(
    file: UploadFile = File(...),
    strength: str = Form(...)
):
    try:
        import tempfile
        from pydub import AudioSegment


        # -------------------------------
        # 📁 SAVE INPUT FILE
        # -------------------------------
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        input_path = os.path.join(UPLOAD_DIR, unique_name)


        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)


        # -------------------------------
        # 🎧 LOAD AUDIO
        # -------------------------------
        y, sr = librosa.load(input_path, sr=None)


        # -------------------------------
        # 🔍 NOISE PROFILE (SMART DETECTION)
        # -------------------------------
        noise_sample = y[0:int(0.5 * sr)]  # first 0.5 sec = noise


        # -------------------------------
        # 🎚️ STRENGTH CONTROL
        # -------------------------------
        if strength == "Low":
            prop_decrease = 0.5
        elif strength == "Medium":
            prop_decrease = 0.8
        else:  # High
            prop_decrease = 1.0


        # -------------------------------
        # 🔥 STEP 1: NOISE REDUCTION
        # -------------------------------
        reduced_noise = nr.reduce_noise(
            y=y,
            sr=sr,
            y_noise=noise_sample,
            prop_decrease=prop_decrease
        )


        # -------------------------------
        # 🔊 STEP 2: BANDPASS FILTER (VOICE FOCUS)
        # -------------------------------
        reduced_noise = librosa.effects.preemphasis(reduced_noise, coef=0.97)


        # -------------------------------
        # 🔊 STEP 3: VOLUME NORMALIZATION
        # -------------------------------
        reduced_noise = reduced_noise / np.max(np.abs(reduced_noise))


        # -------------------------------
        # 🔊 STEP 4: LOUDNESS BOOST
        # -------------------------------
        reduced_noise = reduced_noise * 2.0


        # -------------------------------
        # 💾 SAVE TEMP WAV
        # -------------------------------
        temp_wav = os.path.join(OUTPUT_DIR, f"temp_{unique_name}.wav")
        sf.write(temp_wav, reduced_noise, sr)


        # -------------------------------
        # 🎵 CONVERT TO MP3
        # -------------------------------
        output_filename = f"cleaned_{unique_name}.mp3"
        output_path = os.path.join(OUTPUT_DIR, output_filename)


        audio = AudioSegment.from_wav(temp_wav)
        audio.export(output_path, format="mp3", bitrate="192k")


        # Remove temp file
        os.remove(temp_wav)


        # Remove input file
        if os.path.exists(input_path):
            os.remove(input_path)


        # -------------------------------
        # ✅ RETURN RESPONSE
        # -------------------------------
        return {
            "cleaned_url": f"http://127.0.0.1:8000/download/{output_filename}"
        }


    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)