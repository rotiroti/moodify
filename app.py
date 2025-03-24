#!/usr/bin/env python3

import json
from datetime import datetime

import gradio as gr
import librosa
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from transformers import pipeline

from constants import stylesheet, text_examples
from fusion import AverageFusion, WeightedFusion

# Disable tokenizer parallelism
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open("config.json") as config_file:
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

# Global state to store predictions
EMOTION_STATE = {"Speech": [], "Text": [], "Face": []}

fusion_strategies = {
    "Average": AverageFusion(),
    "Weighted": WeightedFusion({"Speech": 0.3, "Text": 0.2, "Face": 0.5}),
}


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


def fuse_results(strategy_name):
    """Fuse emotions from different modalities using the latest predictions."""
    latest_predictions = {}

    # Get latest prediction for each modality
    for modality in EMOTION_STATE:
        if EMOTION_STATE[modality]:
            latest = sorted(EMOTION_STATE[modality], key=lambda x: x["timestamp"])[-1]
            latest_predictions[modality] = latest["scores"]

    if not latest_predictions:
        return (
            gr.update(
                value="No predictions yet. Please use Speech, Text, or Face tabs to detect emotions first.",
                visible=True,
            ),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    strategy = fusion_strategies[strategy_name]
    top_emotion, final_scores = strategy.fuse(latest_predictions)

    image_path = config["assets"].get(top_emotion, None)
    playlist_html = search_playlist(top_emotion)

    return (
        gr.update(value=final_scores, visible=True),
        gr.update(value=image_path, visible=True),
        gr.update(value=playlist_html, visible=True),
    )


def get_confidences(prediction, labels):
    result = {emotion: 0.0 for emotion in set(labels.values())}
    confidences = {p["label"]: float(p["score"]) for p in prediction}

    for label, score in confidences.items():
        if label in labels:
            mapped_label = labels[label]
            result[mapped_label] = result.get(mapped_label, 0.0) + score

    return result


def ser_predict(inp):
    if inp is None:
        raise gr.Error(
            "No audio detected! Please record your voice or upload an audio file."
        )

    sr, y = inp

    # Mono conversion if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Convert to float if necessary
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32) / np.iinfo(y.dtype).max

    # Resample to target sample rate
    if sr != config["pipelines"]["ser"]["sample_rate"]:
        y = librosa.resample(
            y, orig_sr=sr, target_sr=config["pipelines"]["ser"]["sample_rate"]
        )
        sr = config["pipelines"]["ser"]["sample_rate"]

    # Normalize audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    prediction = ser_pipeline({"sampling_rate": sr, "raw": y})
    confidences = get_confidences(prediction, config["pipelines"]["ser"]["mapping"])
    EMOTION_STATE["Speech"].append({"scores": confidences, "timestamp": datetime.now()})

    return confidences


def ter_predict(inp):
    if not inp or inp.strip() == "":
        raise gr.Error(
            "Empty text! Please write something or choose from the examples below."
        )
    prediction = ter_pipeline(inp, top_k=None)
    confidences = get_confidences(prediction, config["pipelines"]["ter"]["mapping"])
    EMOTION_STATE["Text"].append({"scores": confidences, "timestamp": datetime.now()})

    return confidences


def fer_predict(inp):
    if inp is None:
        raise gr.Error(
            "No image detected! Please take a photo or upload an image showing a face."
        )

    prediction = fer_pipeline(inp, top_k=None)
    confidences = get_confidences(prediction, config["pipelines"]["fer"]["mapping"])
    EMOTION_STATE["Face"].append({"scores": confidences, "timestamp": datetime.now()})

    return confidences


home_tab = gr.Blocks()

with home_tab:
    gr.Markdown(
        """
    # 🎵 Welcome to Moodify! 🎭

    **Moodify** is a multimodal emotion recognition system that analyzes emotions from **Speech, Text, and Facial Expressions** and suggests **personalized Spotify playlists** based on the detected mood!

    ## **🎯 How It Works**
    Moodify combines the power of **deep learning** to recognize emotions from different modalities:

    ### **1. Choose an Input Method**
    - 🎙️ **Speech Emotion Recognition (SER)** → Record or upload an audio file, analyzed using **OpenAI Whisper**
    - 📝 **Text Emotion Recognition (TER)** → Enter a sentence or choose from pre-written examples in **English**, analyzed using **DistilRoBERTa**
    - 😊 **Face Emotion Recognition (FER)** → Capture a photo or upload an image, analyzed using **Vision Transformer (ViT)**

    ### **2. Get Individual Predictions**
    - Each tab (**Speech, Text, Face**) provides an independent emotion prediction
    - Predictions are **automatically stored** for fusion

    ### **3. Generate a Spotify Playlist**
    - Navigate to the **"Spotify Playlist"** tab
    - Choose a **fusion strategy** to combine multiple emotion predictions:
        - **Averaging** → Equal weight for all modalities
        - **Weighted Sum** → Default: Speech (30%), Text (20%), Face (50%)
    - Click **"Find My Playlist"** to generate **music recommendations** based on detected emotions

    ## **💡 Tips for Best Experience**
    - Try different modalities to **enhance emotion detection**
    - Experiment with **fusion strategies** to see how they affect recommendations
    - Click on a playlist to **open it directly in Spotify** and start listening!

    🚀 **Ready to explore music that reflects emotions? Get started now!** 🎶
    """
    )

ser_tab = gr.Interface(
    fn=ser_predict,
    inputs=gr.Audio(type="numpy", format="wav", show_label=False),
    outputs=gr.Label(show_label=False),
    title="Speech Emotion Recognition",
    flagging_mode="never",
)

ter_tab = gr.Interface(
    fn=ter_predict,
    inputs=gr.Textbox(lines=8, show_label=False, placeholder="Enter text here"),
    outputs=gr.Label(show_label=False),
    title="Text-Based Emotion Recognition",
    flagging_mode="never",
    examples_per_page=5,
    examples=text_examples,
)

fer_tab = gr.Interface(
    fn=fer_predict,
    inputs=gr.Image(type="pil", show_label=False),
    outputs=gr.Label(show_label=False),
    title="Face Emotion Recognition",
    flagging_mode="never",
)


def update_weights(speech_w, text_w, face_w):
    """Update weights for weighted fusion."""
    total = speech_w + text_w + face_w
    if total > 0:
        weights = {
            "Speech": speech_w / total,
            "Text": text_w / total,
            "Face": face_w / total,
        }
        fusion_strategies["Weighted"] = WeightedFusion(weights)
    return f"Current weights - Speech: {speech_w/total:.2f}, Text: {text_w/total:.2f}, Face: {face_w/total:.2f}"


playlist_tab = gr.Blocks()

with playlist_tab:
    with gr.Row():
        strategy_selector = gr.Radio(
            choices=list(fusion_strategies.keys()),
            value="Average",
            label="Late Fusion Strategy",
            info="Select fusion method: Average (equal weights) or Weighted (custom weights)",
        )

    with gr.Column(visible=False) as weight_controls:
        weight_info = gr.Markdown("Adjust weights for each modality")
        with gr.Row():
            speech_weight = gr.Slider(
                minimum=0, maximum=1, value=0.3, step=0.1, label="Speech Weight"
            )
            text_weight = gr.Slider(
                minimum=0, maximum=1, value=0.2, step=0.1, label="Text Weight"
            )
            face_weight = gr.Slider(
                minimum=0, maximum=1, value=0.5, step=0.1, label="Face Weight"
            )

    with gr.Row():
        fuse_button = gr.Button("Find My Playlist")
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

    # Show/hide weight controls based on strategy selection
    strategy_selector.change(
        fn=lambda x: gr.update(visible=(x == "Weighted")),
        inputs=[strategy_selector],
        outputs=[weight_controls],
    )

    # Update weights when sliders change
    for slider in [speech_weight, text_weight, face_weight]:
        slider.change(
            fn=update_weights,
            inputs=[speech_weight, text_weight, face_weight],
            outputs=[weight_info],
        )

    # Fusion button click handler
    fuse_button.click(
        fn=fuse_results,
        inputs=[strategy_selector],
        outputs=[final_emotion, html_image, spotify_playlist],
    )


demo = gr.Blocks(
    theme=gr.themes.Citrus(
        font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"]
    ),
    css=stylesheet,
    title="🎭 Moodify | Feel It, Play It",
)

with demo:
    gr.TabbedInterface(
        [home_tab, ser_tab, ter_tab, fer_tab, playlist_tab],
        tab_names=["Home", "Speech", "Text", "Face", "Spotify Playlist"],
        title="🗣️ 📝 😊 Moodify 📊 🎵 🎧",
    )

if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)
