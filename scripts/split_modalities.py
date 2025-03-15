#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import torchaudio
from moviepy import VideoFileClip
from tqdm import tqdm
from transformers import pipeline


def extract_audio(src_path: Path, dst_dir: Path) -> Path:
    """Extract audio from a video file and save it as a WAV file.

    Args:
        src_path: Path to the input video file.
        dst_dir: Path to the output directory where the audio will be saved.

    Returns:
        Path to the saved audio file.
    """
    clip = VideoFileClip(src_path)
    dst_path = dst_dir / src_path.with_suffix(".wav").name
    clip.audio.write_audiofile(dst_path, codec="pcm_s16le", fps=48000, ffmpeg_params=["-ac", "1"])

    print(f"Audio saved to {dst_path}")

    return dst_path


def extract_text(src_path: Path, asr_pipeline: pipeline, dst_dir: Path):
    """Extract text from an audio file using wav2vec2 Hugging Face pipeline.

    Args:
        src_path: Path to the input audio file.
        dst_dir: Path to the output directory where the transcript will be saved.
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(str(src_path))

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    # Resample to 16kHz (required by Wav2Vec2)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    waveform = waveform.numpy().squeeze()
    transcript = asr_pipeline({"sampling_rate": 16000, "raw": waveform})["text"]

    if transcript:
        transcript = transcript.lower()
        dst_path = dst_dir / src_path.with_suffix(".txt").name
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"Transcript saved to {dst_path}")


def extract_frames(src_path: Path, dst_dir: Path, fps: int = 5):
    """Extract frames from a video file at a given frame rate.

    Args:
        src_path: Path to the input video file.
        dst_dir: Path to the output directory where frames will be saved.
        fps: Number of frames per second to extract from the video.
    """
    cap = cv2.VideoCapture(str(src_path))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, video_fps // fps)
    frame_count = 0
    saved_frames = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame_filename = dst_dir / f"{src_path.stem}_frame_{frame_count}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_frames += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Extracted {frame_count} frames")


def split_modalities(src_path: Path, asr_pipeline: pipeline, dst_dir: Path, fps: int = 5):
    """Split a video file into its modalities: audio, text, and frames.

    Args:
        video_path: Path to the input video file.
        dst_dir: Path to the output directory where modalities will be saved.
        fps: Number of frames per second to extract from the video.
    """
    tqdm.write(f"Processing {src_path.name}...")
    video_dst = dst_dir / src_path.stem
    video_dst.mkdir(parents=True, exist_ok=True)

    audio_path = extract_audio(src_path, video_dst)
    extract_text(audio_path, asr_pipeline, video_dst)
    extract_frames(src_path, video_dst, fps)


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="Path to the input video")
    parser.add_argument("dst", type=Path, help="Path to the output directory")
    parser.add_argument("--fps", type=int, default=5, help="Number of frames per second to extract from the video")
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    # Instantiate wav2vec2
    wav2vec2 = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h")

    if args.src.is_dir():
        for video_file in tqdm(list(args.src.glob("*.mp4")), desc="Processing Videos", unit="video"):
            split_modalities(video_file, wav2vec2, args.dst, args.fps)
    else:
        split_modalities(args.src, wav2vec2, args.dst, args.fps)


if __name__ == "__main__":
    run()
