stylesheet = """
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

text_examples = [
    "I can't stand it when people don't follow the rules!",
    "The smell of rotten food makes me feel sick.",
    "Walking alone in the dark really makes me uneasy.",
    "I love playing the guitar, it makes me feel free and happy!",
    "Since you left, the house feels empty and quiet.",
]
