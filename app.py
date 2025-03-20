#!/usr/bin/env python3

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import pipeline

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

LOGFILE = "moodify.csv"

MODELS_NAME = {
    "ser": "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
    "ter": "michellejieli/emotion_text_classifier",
    # "ter": "j-hartmann/emotion-english-distilroberta-base"
    "fer": "dima806/facial_emotions_image_detection",
}

ser_pipeline = pipeline("audio-classification", model=MODELS_NAME["ser"])
ter_pipeline = pipeline("text-classification", model=MODELS_NAME["ter"])
fer_pipeline = pipeline("image-classification", model=MODELS_NAME["fer"], use_fast=True)
emotion_labels = ["angry", "disgust", "fear", "happy", "sad"]


def ser_predict(inp):
    sr, y = inp

    # Mono conversion if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    # Resample to 16kHz
    target_sr = 16000
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    prediction = ser_pipeline({"sampling_rate": sr, "raw": y})
    confidences = {prediction[i]['label']: float(prediction[i]['score']) for i in range(len(emotion_labels))}

    return confidences

def ter_predict(inp):
    prediction = ter_pipeline(inp, top_k=None)
    confidences = {pred['label']: float(pred['score']) for pred in prediction}

    return confidences

def fer_predict(inp):
    preidction = fer_pipeline(inp)
    confidences = {pred['label']: float(pred['score']) for pred in preidction}

    return confidences

ser_tab = gr.Interface(
    fn=ser_predict,
    inputs=gr.Audio(type="numpy", format="wav"),
    outputs=gr.Label(),
    title="Speech Emotion Recognition",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=LOGFILE,
    )
)

ter_tab = gr.Interface(
    fn=ter_predict,
    inputs=gr.Textbox(lines=5, label="Enter your text here"),
    outputs=gr.Label(),
    title="Text-Based Emotion Recognition",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=LOGFILE,
    )
)

fer_tab = gr.Interface(
    fn=fer_predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Facial Emotion Recognition",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=LOGFILE,
    )
)

demo = gr.Blocks(theme=gr.themes.Ocean())

with demo:
    gr.TabbedInterface([ser_tab, ter_tab, fer_tab], tab_names=["Speech", "Text", "Face"], title="Moodify")

if __name__ == '__main__':
    demo.launch(debug=True, show_error=True)
