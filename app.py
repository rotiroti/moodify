#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import gradio as gr
import librosa
import numpy as np
import pandas as pd
import spotipy
from dotenv import dotenv_values
from spotipy.oauth2 import SpotifyClientCredentials
from transformers import pipeline

LOGFILE = "moodify.csv"
TARGET_SAMPLE_RATE = 16000
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness"]
CONFIG = {
    "services": {
        "spotify": {},
    },
    "assets": {
        "joy": "./assets/joy.gif",
        "disgust": "./assets/disgust.gif",
        "fear": "./assets/fear.gif",
        "anger": "./assets/anger.gif",
        "sadness": "./assets/sadness.gif",
    },
    "pipelines": {
        "ser": {
            "task": "audio-classification",
            "model": "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
            "mapping": {
                "angry": "anger",
                "disgust": "disgust",
                "fearful": "fear",
                "happy": "joy",
                "sad": "sadness",
            },
        },
        "ter": {
            "task": "text-classification",
            "model": "michellejieli/emotion_text_classifier",
            "mapping": {
                "anger": "anger",
                "disgust": "disgust",
                "fear": "fear",
                "joy": "joy",
                "sadness": "sadness",
            },
        },
        "fer": {
            "task": "image-classification",
            "model": "dima806/facial_emotions_image_detection",
            "mapping": {
                "angry": "anger",
                "disgust": "disgust",
                "fear": "fear",
                "happy": "joy",
                "sad": "sadness",
            },
        },
    },
}
CONFIG["services"]["spotify"] |= dotenv_values(".env")

ser_pipeline = pipeline(
    CONFIG["pipelines"]["ser"]["task"],
    model=CONFIG["pipelines"]["ser"]["model"],
    use_fast=True,
    trust_remote_code=True,
)

ter_pipeline = pipeline(
    CONFIG["pipelines"]["ter"]["task"],
    model=CONFIG["pipelines"]["ter"]["model"],
    use_fast=True,
    trust_remote_code=True,
)

fer_pipeline = pipeline(
    CONFIG["pipelines"]["fer"]["task"],
    model=CONFIG["pipelines"]["fer"]["model"],
    use_fast=True,
    trust_remote_code=True,
)

# Spotify Client Configuration
spotify_client = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CONFIG["services"]["spotify"]["SPOTIPY_CLIENT_ID"],
        client_secret=CONFIG["services"]["spotify"]["SPOTIPY_CLIENT_SECRET"],
    )
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


def search_playlist(emotion: str):
    query = emotion.capitalize() + " Mood"
    response = spotify_client.search(q=query, limit=10, type="playlist")

    if not response or "playlists" not in response:
        return "<p>No playlists found for this emotion.</p>"

    playlists = response.get("playlists", {}).get("items", [])
    valid_playlists = [p for p in playlists if p is not None]

    if not valid_playlists:
        return "<p style='font-size: 1.2em; color: #ff4b4b; text-align: center;'>No playlists available.</p>"

    html_content = """
    <div class="playlist-container">
        <h3 class="playlist-title">Suggested Playlists</h3>
        <div class="playlist-grid">
    """

    for playlist in valid_playlists:
        image_url = (
            playlist["images"][0]["url"]
            if playlist["images"]
            else "https://via.placeholder.com/300"
        )
        name = playlist["name"]
        playlist_url = playlist["external_urls"]["spotify"]
        track_count = playlist["tracks"]["total"]
        track_text = "1 track" if track_count == 1 else f"{track_count} tracks"

        html_content += f"""
        <div class="playlist-card">
            <div class="playlist-img-container">
                <img class="playlist-img" src="{image_url}" alt="{name}">
                <a href="{playlist_url}" target="_blank" class="play-button">
                    <div class="play-icon"></div>
                </a>
            </div>
            <div class="playlist-name">{name}</div>
            <div class="playlist-tracks">{track_text}</div>
            <a href="{playlist_url}" target="_blank" class="spotify-link">
                <span class="spotify-icon">🎵</span> Open in Spotify
            </a>
        </div>
        """

    html_content += """
        </div>
    </div>
    """

    return html_content


def fuse_results():
    confidence_file = _parse_confidence_file()
    if confidence_file is None:
        return "No data available"

    top_emotion, avg_scores = _average_fusion(confidence_file)
    image_path = CONFIG["assets"].get(top_emotion, None)
    playlist_html = search_playlist(top_emotion)

    return (
        gr.update(value=avg_scores, visible=True),
        gr.update(value=image_path, visible=True),
        gr.update(value=playlist_html, visible=True),
    )


def ser_predict(inp):
    preprocessed = _preprocess_audio(inp)
    raw_predictions = ser_pipeline(preprocessed)
    confidences = _compute_confidences(
        raw_predictions, CONFIG["pipelines"]["ser"]["mapping"]
    )

    return gr.update(value=confidences, visible=True)


def ter_predict(inp):
    raw_predictions = ter_pipeline(inp, top_k=None)
    confidences = _compute_confidences(
        raw_predictions, CONFIG["pipelines"]["ter"]["mapping"]
    )

    return gr.update(value=confidences, visible=True)


def fer_predict(inp):
    raw_predictions = fer_pipeline(inp)
    confidences = _compute_confidences(
        raw_predictions, CONFIG["pipelines"]["fer"]["mapping"]
    )

    return gr.update(value=confidences, visible=True)


ser_tab = gr.Interface(
    fn=ser_predict,
    inputs=gr.Audio(type="numpy", format="wav", show_label=False),
    outputs=gr.Label(show_label=False, visible=False),
    title="Speech Emotion Recognition",
    flagging_mode="auto",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=LOGFILE,
    ),
)

ter_tab = gr.Interface(
    fn=ter_predict,
    inputs=gr.Textbox(lines=10, show_label=False, placeholder="Enter text here"),
    outputs=gr.Label(show_label=False, visible=False),
    title="Text-Based Emotion Recognition",
    flagging_mode="auto",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=LOGFILE,
    ),
    examples_per_page=25,
    examples=[
        "Non sopporto quando le persone non rispettano le regole!",
        "I can't stand it when people don't follow the rules!",
        "¡No soporto cuando la gente no respeta las reglas!",
        "L'odore di cibo avariato mi fa venire la nausea.",
        "The smell of rotten food makes me feel sick.",
        "El olor a comida podrida me da náuseas.",
        "Camminare da solo nel buio mi mette davvero a disagio.",
        "Walking alone in the dark really makes me uneasy.",
        "Caminar solo en la oscuridad realmente me pone nervioso.",
        "Adoro suonare la chitarra, mi fa sentire libero e felice!",
        "I love playing the guitar, it makes me feel free and happy!",
        "¡Me encanta tocar la guitarra, me hace sentir libre y feliz!",
        "Da quando te ne sei andato, la casa sembra vuota e silenziosa.",
        "Since you left, the house feels empty and quiet.",
        "Desde que te fuiste, la casa se siente vacía y silenciosa.",
    ],
)

fer_tab = gr.Interface(
    fn=fer_predict,
    inputs=gr.Image(type="pil", show_label=False),
    outputs=gr.Label(show_label=False, visible=False),
    title="Facial Emotion Recognition",
    flagging_mode="auto",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=LOGFILE,
    ),
)

playlist_tab = gr.Blocks()

with playlist_tab:
    with gr.Row():
        fuse_button = gr.Button("Merge Modalities")
    with gr.Row():
        final_emotion = gr.Label(show_label=False, visible=False)
        html_image = gr.Image(
            show_download_button=False,
            show_fullscreen_button=False,
            show_share_button=False,
            show_label=False,
            visible=False,
        )
    with gr.Row():
        spotify_playlist = gr.HTML(visible=False)
    fuse_button.click(
        fn=fuse_results,
        inputs=[],
        outputs=[final_emotion, html_image, spotify_playlist],
    )

css = """
:root {
    --spotify-green: #1DB954;
    --spotify-green-dark: #17A74B;
    --spotify-black: #121212;
    --spotify-dark-gray: #212121;
    --spotify-light-gray: #B3B3B3;
    --card-bg-light: #ffffff;
    --card-bg-dark: #181818;
    --text-primary-light: #000000;
    --text-primary-dark: #ffffff;
    --text-secondary-light: #6a6a6a;
    --text-secondary-dark: #a7a7a7;
    --hover-light: #f7f7f7;
    --hover-dark: #282828;
}

.gradio-container.dark {
    --card-bg: var(--card-bg-dark);
    --text-primary: var(--text-primary-dark);
    --text-secondary: var(--text-secondary-dark);
    --hover-bg: var(--hover-dark);
}

.gradio-container:not(.dark) {
    --card-bg: var(--card-bg-light);
    --text-primary: var(--text-primary-light);
    --text-secondary: var(--text-secondary-light);
    --hover-bg: var(--hover-light);
}

.playlist-container {
    font-family: 'Circular', 'Gotham', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    max-width: 100%;
    margin: 0 auto;
    color: var(--text-primary);
}

.playlist-title {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 24px;
    color: var(--text-primary);
    text-align: center;
}

.playlist-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 24px;
    margin-top: 20px;
    width: 100%;
}

.playlist-card {
    background-color: var(--card-bg);
    border-radius: 8px;
    padding: 16px;
    transition: background-color 0.3s ease;
    cursor: pointer;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

.playlist-card:hover {
    background-color: var(--hover-bg);
    transform: translateY(-4px);
}

.playlist-img-container {
    position: relative;
    width: 100%;
    margin-bottom: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
}

.playlist-img {
    width: 100%;
    aspect-ratio: 1/1;
    border-radius: 8px;
    object-fit: cover;
}

.play-button {
    position: absolute;
    bottom: 8px;
    right: 8px;
    background-color: var(--spotify-green);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    transition: all 0.3s ease;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
}

.playlist-card:hover .play-button {
    opacity: 1;
    transform: translateY(-4px);
}

.play-icon {
    width: 0;
    height: 0;
    border-top: 8px solid transparent;
    border-bottom: 8px solid transparent;
    border-left: 12px solid white;
    margin-left: 3px;
}

.playlist-name {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: var(--text-primary);
}

.playlist-tracks {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-bottom: 12px;
}

.spotify-link {
    color: var(--spotify-green);
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    transition: color 0.2s ease;
}

.spotify-link:hover {
    color: var(--spotify-green-dark);
}

.spotify-icon {
    display: inline-block;
    margin-right: 4px;
    font-size: 1.2em;
}

@media (max-width: 768px) {
    .playlist-grid {
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 16px;
    }
}

footer {
    display: none !important;
}
"""


demo = gr.Blocks(theme=gr.themes.Ocean(), css=css)

with demo:
    gr.TabbedInterface(
        [ser_tab, ter_tab, fer_tab, playlist_tab],
        tab_names=["Speech", "Text", "Facial", "Spotify Playlist"],
        title="Moodify",
    )

if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
