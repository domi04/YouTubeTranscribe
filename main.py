import argparse
from pytubefix import YouTube
from pytubefix.cli import on_progress
import mlx_whisper


def main(url, model):
    yt = YouTube(url, on_progress_callback=on_progress)
    print(yt.title)

    ys = yt.streams.get_audio_only()

    ys.download(filename="yt_audio.wav")

    print("Transcribing audio to text...")
    model_path = (
        "mlx-community/whisper-tiny"
        if model == "tiny"
        else "mlx-community/whisper-large-v3-turbo"
    )
    text = mlx_whisper.transcribe("yt_audio.wav", path_or_hf_repo=model_path).get(
        "text"
    )

    with open("yt_audio.txt", "w") as f:
        f.write(text)

    print("Transcription saved to yt_audio.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe YouTube video audio to text."
    )
    parser.add_argument("url", help="The URL of the YouTube video to transcribe.")
    parser.add_argument(
        "--model",
        choices=["tiny", "large"],
        default="tiny",
        help="The model to use for transcription. Options are tiny, large (default: tiny).",
    )
    args = parser.parse_args()
    main(args.url, args.model)
