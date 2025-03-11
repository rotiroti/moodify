#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import speech_recognition as sr
from moviepy import VideoFileClip


def extract_audio_from_video(src_path: Path, dst_dir: Path) -> Path:
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


def extract_text(src_path, dst_dir: Path):
    """Extract text from an audio file using Google Speech Recognition.

    Args:
        src_path: Path to the input audio file.
        dst_dir: Path to the output directory where the transcript will be saved.
    """
    recognizer = sr.Recognizer()
    transcript = None

    with sr.AudioFile(str(src_path)) as src:
        audio = recognizer.record(src)
    try:
        transcript = recognizer.recognize_google(audio)
        print(f"Transcript: {transcript}")
    except sr.UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError:
        print("Could not request results from the speech recognition service")

    if transcript:
        dst_path = dst_dir / src_path.with_suffix(".txt").name
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"Transcript saved to {dst_path}")


def extract_frames(src_path, dst_dir: Path, fps: int = 5):
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


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="Path to the input video")
    parser.add_argument("dst", type=Path, help="Path to the output directory")
    parser.add_argument("--fps", type=int, default=5, help="Number of frames per second to extract from the video")
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    audio_path = extract_audio_from_video(args.src, args.dst)
    extract_text(audio_path, args.dst)
    extract_frames(args.src, args.dst, args.fps)


if __name__ == "__main__":
    run()
