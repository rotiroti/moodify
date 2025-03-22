#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import gradio as gr
import librosa
import numpy as np
import pandas as pd
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


def _parse_confidence_file():
    confidence_dir = Path(".gradio/flagged")

    if not confidence_dir.exists():
        return None

    confidence_file = confidence_dir / LOGFILE
    df = pd.read_csv(confidence_file)

    # Remove invalid entries (rows with `"label": null, "confidences": null"`)
    df = df[~df["output"].str.contains('"label": null', na=False)]

    # Identify modality based on 'inp' column
    # TODO: Review valid extensions for each speech modality
    df["modality"] = df["inp"].apply(
        lambda x: (
            "Speech"
            if isinstance(x, str) and x.endswith(".wav")
            else "Face"
            if isinstance(x, str) and x.endswith((".jpg", ".png"))
            else "Text"
        )
    )

    # Keep only the last occurrence of each modality
    df = df.groupby("modality").last().reset_index()

    results = {}

    for _, row in df.iterrows():
        parsed_json = json.loads(row["output"].replace('""', '"'))
        results[row["modality"]] = {
            "emotion": parsed_json["label"],
            "scores": {
                entry["label"]: float(entry["confidence"])
                for entry in parsed_json["confidences"]
            },
            "timestamp": row["timestamp"],
        }

    return results


def _compute_scores_matrix(confidence_file):
    # Find the first available modality to get emotion labels
    for modality in ["Face", "Speech", "Text"]:
        if modality in confidence_file:
            emotion_labels = list(confidence_file[modality]["scores"].keys())
            break
    else:
        raise ValueError("No valid modality found in confidence file.")

    scores_matrix = np.array(
        [list(mod["scores"].values()) for mod in confidence_file.values()]
    )

    return scores_matrix, emotion_labels


def _average_fusion(confidence_file):
    scores_matrix, emotion_labels = _compute_scores_matrix(confidence_file)

    # Normalize scores before averaging
    scores_matrix = scores_matrix / scores_matrix.sum(axis=1, keepdims=True)
    avg_scores = np.mean(scores_matrix, axis=0)
    top_emotion = emotion_labels[np.argmax(avg_scores)]

    return top_emotion, dict(zip(emotion_labels, avg_scores))


def generate_playlist():
    confidence_file = _parse_confidence_file()
    if confidence_file is None:
        return "No data available"

    top_emotion, avg_scores = _average_fusion(confidence_file)

    return avg_scores

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

playlist_tab = gr.Interface(
    fn=generate_playlist,
    inputs=[],
    outputs=gr.Label(),
    flagging_mode="never",
)

demo = gr.Blocks(theme=gr.themes.Ocean())

with demo:
    gr.TabbedInterface(
        [ser_tab, ter_tab, fer_tab, playlist_tab],
        tab_names=["Speech", "Text", "Face", "Generate Playlist"],
        title="Moodify",
    )

if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
