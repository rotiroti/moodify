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

## License

The Moodify project is licensed under the terms of the MIT license.
