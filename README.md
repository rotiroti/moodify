# Moodify

**Moodify** is a multimodal emotion recognition application that analyzes emotions from speech, text, and facial expressions. The system combines these predictions using different fusion strategies and suggests personalized music playlists via Spotify API that match your emotional state.

## Requirements

- Python 3.10 or higher
- Spotify Developer Account for API access
- uv package manager

## Installation

1. Install uv:

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Clone the repository:

    ```shell
    git clone https://github.com/rotiroti/msc/moodify.git
    cd moodify
    ```

3. Create and activate a virtual environment:

    ```shell
    uv venv
    source .venv/bin/activate
    ```

4. Install dependencies using uv:

    ```shell
    uv sync
    ```

## Configuration

1. Copy the example configuration file:

    ```shell
    cp config.json.example config.json
    ```

2. Get your Spotify API credentials:
    - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
    - Create a new application
    - Copy your Client ID and Client Secret

3. Update `config.json` with your Spotify credentials:

    ```json
    {
      "services": {
        "spotify": {
          "client_id": "your_client_id",
          "client_secret": "your_client_secret"
        }
      }
    }
    ```

## Running the Application

1. Activate the virtual environment (if not already activated):

    ```shell
    source .venv/bin/activate
    ```

2. Start the application:

    ```shell
    uv run python app.py
    ```

3. Open your browser and navigate to:

    ```shell
    http://localhost:7860
    ```

## Models

This project uses the following pre-trained models for emotion recognition:

### Speech Emotion Recognition

- **Model**: [Speech Emotion Recognition with OpenAI Whisper Large v3](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3)
- **Architecture**: OpenAI Whisper Large v3
- **Reference**: [Radford et al. (2022) - Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)

### Text Emotion Recognition

- **Model**: [Emotion English DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/)
- **Architecture**: DistilRoBERTa-base
- **References**: [Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.](https://arxiv.org/abs/1910.01108)

### Facial Emotion Recognition

- **Model**: [Facial Emotions Image Detection](https://huggingface.co/dima806/facial_emotions_image_detection)
- **Architecture**: Vision Transformer (ViT)
- **Reference**: [Dosovitskiy et al. (2021) - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

Each model was chosen for its performance and compatibility with our emotion categories. The models are accessed through the Hugging Face Transformers library.

## License

The Moodify project is licensed under the terms of the MIT license.
