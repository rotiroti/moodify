#!/usr/bin/env python3

import json
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from transformers import pipeline

from constants import stylesheet, text_examples

TARGET_SAMPLE_RATE = 16000

with open("config.json", "r") as config_file:
    config = json.load(config_file)

ser_pipeline = pipeline(
    config["pipelines"]["ser"]["task"],
    model=config["pipelines"]["ser"]["model"],
    use_fast=True,
    trust_remote_code=True,
)

ter_pipeline = pipeline(
    config["pipelines"]["ter"]["task"],
    model=config["pipelines"]["ter"]["model"],
    use_fast=True,
    trust_remote_code=True,
)

fer_pipeline = pipeline(
    config["pipelines"]["fer"]["task"],
    model=config["pipelines"]["fer"]["model"],
    use_fast=True,
    trust_remote_code=True,
)

# Spotify Client Configuration
spotify_client = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=config["services"]["spotify"]["client_id"],
        client_secret=config["services"]["spotify"]["client_secret"],
    )
)


def _prediction_to_dict(predictions):
    if isinstance(predictions, list):
        return {pred["label"]: float(pred["score"]) for pred in predictions}
    return {label: float(score) for label, score in predictions.items()}


def _from_models_to_emotion_labels(emotion_dict, mapping_dict):
    result = {}

    for label, score in emotion_dict.items():
        if label in mapping_dict:
            mapped_label = mapping_dict[label]
            result[mapped_label] = result.get(mapped_label, 0.0) + score
    return result


def _filter_emotions(mapped_emotions):
    result = {
        emotion: mapped_emotions.get(emotion, 0.0) for emotion in config["labels"]
    }
    total = sum(result.values())

    if total > 0:
        for emotion in result:
            result[emotion] /= total
    return result


def _compute_confidences(raw_predictions, model_mapping):
    emotion_dict = _prediction_to_dict(raw_predictions)
    mapped_emotions = _from_models_to_emotion_labels(emotion_dict, model_mapping)
    return _filter_emotions(mapped_emotions)


def _preprocess_audio(inp):
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

    confidence_file = confidence_dir / config["logfile"]
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
    image_path = config["assets"].get(top_emotion, None)
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
        raw_predictions, config["pipelines"]["ser"]["mapping"]
    )

    return confidences


def ter_predict(inp):
    raw_predictions = ter_pipeline(inp, top_k=None)
    confidences = _compute_confidences(
        raw_predictions, config["pipelines"]["ter"]["mapping"]
    )

    return confidences


def fer_predict(inp):
    raw_predictions = fer_pipeline(inp)
    confidences = _compute_confidences(
        raw_predictions, config["pipelines"]["fer"]["mapping"]
    )

    return confidences


ser_tab = gr.Interface(
    fn=ser_predict,
    inputs=gr.Audio(type="numpy", format="wav", show_label=False),
    outputs=gr.Label(show_label=False),
    title="Speech Emotion Recognition",
    flagging_mode="auto",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=config["logfile"],
    ),
)

ter_tab = gr.Interface(
    fn=ter_predict,
    inputs=gr.Textbox(lines=10, show_label=False, placeholder="Enter text here"),
    outputs=gr.Label(show_label=False),
    title="Text-Based Emotion Recognition",
    flagging_mode="auto",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=config["logfile"],
    ),
    examples_per_page=25,
    examples=text_examples,
)

fer_tab = gr.Interface(
    fn=fer_predict,
    inputs=gr.Image(type="pil", show_label=False),
    outputs=gr.Label(show_label=False),
    title="Facial Emotion Recognition",
    flagging_mode="auto",
    flagging_callback=gr.CSVLogger(
        dataset_file_name=config["logfile"],
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

demo = gr.Blocks(theme=gr.themes.Ocean(), css=stylesheet)

with demo:
    gr.TabbedInterface(
        [ser_tab, ter_tab, fer_tab, playlist_tab],
        tab_names=["Speech", "Text", "Facial", "Spotify Playlist"],
        title="Moodify",
    )

if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
