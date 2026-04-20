# Voice-Lab-Emotion-Detection-FYP
Speech Emotion Recognition, in it you can find the emotion from voice, reduce noise from voice, enhance voice, convert voice to text and text to voice.
🎙️ Multi-Functional Speech Emotion Recognition (SER) System
This project is an end-to-end audio processing pipeline that combines advanced speech enhancement with Deep Learning-based emotion detection. While the core of the project is identifying human emotions from vocal cues, it includes a suite of essential audio tools like Noise Reduction and Voice Enhancement to ensure high accuracy even in real-world, noisy environments.

🚀 Key Features
Our system is divided into five specialized modules:

🧠 Emotion Detection (Core):
The heart of the project. Using a CNN-LSTM (Convolutional Neural Network + Long Short-Term Memory) architecture, the system classifies audio into emotions such as Happy, Sad, Angry, Neutral, Fearful, and Surprised. It extracts MFCCs (Mel-frequency cepstral coefficients) to "hear" the emotional pitch and tone.

🔇 Noise Reduction:
Implements spectral gating and deep filtering to remove background hums, clicks, and environmental noise, ensuring the "cleanest" signal possible for the AI.

🔊 Voice Enhancement:
Boosts the clarity and gain of the speaker’s voice, normalizing the volume and sharpening the frequencies that matter most for speech intelligibility.

🗣️ Voice-to-Text (ASR):
Transcribes the spoken words into written text in real-time, allowing for simultaneous analysis of what is being said alongside how it is being said.

✍️ Text-to-Voice (TTS):
A generative module that can convert written text back into natural-sounding speech, completing the full communication loop.

🛠️ Tech Stack
Language: Python 3.x

Audio Processing: Librosa, PyAudio, SoundFile

Deep Learning: TensorFlow / Keras / PyTorch

Speech Engines: OpenAI Whisper (Speech-to-Text), Google TTS / Coqui TTS (Text-to-Speech)

Denoising: DeepFilterNet or Spectral Gating

📊 How it Works
Input: User records or uploads an audio file (.wav or .mp3).

Preprocessing: The Noise Reduction and Enhancement modules clean the signal.

Analysis:

The Emotion Engine extracts MFCC and Chroma features to predict the mood.

The ASR Engine converts the audio into text.

Output: A detailed report showing the identified emotion, the confidence score, and the transcribed text.
