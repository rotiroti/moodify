#!/usr/bin/env python3

from typing import Any, Dict, List, Tuple, Union

import gradio as gr
import librosa
import numpy as np
from transformers import pipeline

LOGFILE = "moodify.csv"
TARGET_SAMPLE_RATE = 16000
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness"]
CONFIG = {
    "ser": {
        "mapping": {
            "angry": "anger",
            "disgust": "disgust",
            "fearful": "fear",
            "happy": "joy",
            "sad": "sadness",
        },
    },
    "ter": {
        "mapping": {
            "anger": "anger",
            "disgust": "disgust",
            "fear": "fear",
            "joy": "joy",
            "sadness": "sadness",
        },
    },
    "fer": {
        "mapping": {
            "angry": "anger",
            "disgust": "disgust",
            "fear": "fear",
            "happy": "joy",
            "sad": "sadness",
        },
    },
}

# Instantiate the pipelines
ser_pipeline = pipeline(
    "audio-classification",
    model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
    use_fast=True,
    trust_remote_code=True,
)

ter_pipeline = pipeline(
    "text-classification",
    model="michellejieli/emotion_text_classifier",
    use_fast=True,
    trust_remote_code=True,
)

fer_pipeline = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    use_fast=True,
    trust_remote_code=True,
)


def _prediction_to_dict(
    predictions: Union[List[Dict[str, Any]], Dict[str, float]],
) -> Dict[str, float]:
    if isinstance(predictions, list):
        return {pred["label"]: float(pred["score"]) for pred in predictions}
    return {label: float(score) for label, score in predictions.items()}


def _from_models_to_emotion_labels(
    emotion_dict: Dict[str, float], mapping_dict: Dict[str, str]
) -> Dict[str, float]:
    result: Dict[str, float] = {}

    for label, score in emotion_dict.items():
        if label in mapping_dict:
            mapped_label = mapping_dict[label]
            result[mapped_label] = result.get(mapped_label, 0.0) + score
    return result


def _filter_emotions(mapped_emotions: Dict[str, float]) -> Dict[str, float]:
    result = {emotion: mapped_emotions.get(emotion, 0.0) for emotion in EMOTION_LABELS}
    total = sum(result.values())

    if total > 0:
        for emotion in result:
            result[emotion] /= total
    return result


def _compute_confidences(
    raw_predictions: Any, model_mapping: Dict[str, str]
) -> Dict[str, float]:
    """Full processing pipeline for emotion predictions."""
    emotion_dict = _prediction_to_dict(raw_predictions)
    mapped_emotions = _from_models_to_emotion_labels(emotion_dict, model_mapping)
    return _filter_emotions(mapped_emotions)


def _preprocess_audio(inp: Tuple[int, np.ndarray]) -> Dict[str, Any]:
    """Preprocess audio input."""
    sr, y = inp

    # Mono conversion if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Convert to float if necessary
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    # Resample to target sample rate
    if sr != TARGET_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    # Normalize audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return {"sampling_rate": sr, "raw": y}


def ser_predict(inp: Tuple[int, np.ndarray]) -> Dict[str, float]:
    preprocessed = _preprocess_audio(inp)
    raw_predictions = ser_pipeline(preprocessed)
    confidences = _compute_confidences(raw_predictions, CONFIG["ser"]["mapping"])

    return confidences


def ter_predict(inp: str) -> Dict[str, float]:
    raw_predictions = ter_pipeline(inp, top_k=None)
    confidences = _compute_confidences(raw_predictions, CONFIG["ter"]["mapping"])

    return confidences


def fer_predict(inp: Any) -> Dict[str, float]:
    raw_predictions = fer_pipeline(inp)
    confidences = _compute_confidences(raw_predictions, CONFIG["fer"]["mapping"])

    return confidences


ser_tab = gr.Interface(
    fn=ser_predict,
    inputs=gr.Audio(type="numpy", format="wav"),
    outputs=gr.Label(),
    title="Speech Emotion Recognition",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=LOGFILE,
    ),
)

ter_tab = gr.Interface(
    fn=ter_predict,
    inputs=gr.Textbox(lines=5, label="Enter your text here"),
    outputs=gr.Label(),
    title="Text-Based Emotion Recognition",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=LOGFILE,
    ),
)

fer_tab = gr.Interface(
    fn=fer_predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Facial Emotion Recognition",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=LOGFILE,
    ),
)

demo = gr.Blocks(theme=gr.themes.Ocean())

with demo:
    gr.TabbedInterface(
        [ser_tab, ter_tab, fer_tab],
        tab_names=["Speech", "Text", "Face"],
        title="Moodify",
    )

if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
