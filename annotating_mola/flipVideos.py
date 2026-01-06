import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd):
    """Run a command and raise with useful output on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"STDOUT:\n{p.stdout}\n\n"
            f"STDERR:\n{p.stderr}\n"
        )
    return p


def ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    out = run(cmd).stdout
    return json.loads(out)


def pick_video_encoder(video_codec_name: str) -> str:
    # Map common codecs to typical ffmpeg encoders
    # (This is about choosing an encoder, not "copying" codec settings 1:1.)
    if video_codec_name in ("h264", "avc1"):
        return "libx264"
    if video_codec_name in ("hevc", "h265"):
        return "libx265"
    # Fallback: widely compatible MP4 video encoder
    return "libx264"


def main(videos_dir: str):
    videos_dir = Path(videos_dir).resolve()
    if not videos_dir.is_dir():
        raise ValueError(f"Not a directory: {videos_dir}")

    # Ensure ffmpeg exists
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffmpeg/ffprobe not found on PATH. Install them, e.g.:\n"
            "  sudo apt-get update && sudo apt-get install -y ffmpeg"
        )

    out_dir = videos_dir.parent / f"{videos_dir.name}_flipped"
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(videos_dir.glob("*.mp4"))
    if not mp4s:
        print(f"No .mp4 files found in {videos_dir}")
        return

    for in_path in mp4s:
        out_path = out_dir / in_path.name

        probe = ffprobe_json(in_path)
        video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]

        if not video_streams:
            print(f"Skipping (no video stream): {in_path.name}")
            continue

        v = video_streams[0]
        vcodec = v.get("codec_name", "")
        encoder = pick_video_encoder(vcodec)

        # Try to preserve bitrate if present
        # (May be missing or unreliable depending on container/stream.)
        target_bitrate = v.get("bit_rate")

        # Preserve FPS if available
        # Use r_frame_rate, which is usually like "30/1"
        fps = v.get("r_frame_rate")

        # Preserve pixel format if available
        pix_fmt = v.get("pix_fmt")

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(in_path)]

        # Video filter
        cmd += ["-vf", "hflip"]

        # Video encoding choice
        cmd += ["-c:v", encoder]

        # Keep frame rate if we can
        if fps and fps != "0/0":
            cmd += ["-r", fps]

        # Keep pixel format if known (helps keep compatibility/consistency)
        if pix_fmt:
            cmd += ["-pix_fmt", pix_fmt]

        # Try to keep bitrate (optional)
        if target_bitrate and target_bitrate.isdigit():
            cmd += ["-b:v", target_bitrate]

        # Audio: copy if present, else ignore
        if audio_streams:
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-an"]

        # Copy metadata (where possible)
        cmd += ["-map_metadata", "0"]

        # Ensure MP4 is seekable/streamable
        cmd += ["-movflags", "+faststart"]

        cmd += [str(out_path)]

        try:
            run(cmd)
            print(f"OK: {in_path.name} -> {out_path}")
        except Exception as e:
            print(f"FAILED: {in_path.name}\n{e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flip all MP4 videos in a directory horizontally.")
    parser.add_argument("videos_dir", help="Path to directory containing .mp4 videos")
    args = parser.parse_args()
    main(args.videos_dir)
