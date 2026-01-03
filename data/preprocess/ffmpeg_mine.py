import os
import subprocess


import os
import subprocess
import re

def ffmpeg_once_from_frames(
    src_path: str,
    dst_path: str,
    *,
    input_fps: int = 30,   # FPS the frame sequence represents (timeline)
    fps: int = None,       # output FPS (e.g. 2)
    resolution: int = None,
    pad: str = "#000000",
    mode: str = "bicubic",
    pattern: str = "%05d.jpg",
    log_path: str = None,  # if set, saves ffmpeg output log here
    return_chosen_frames: bool = True,
):
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

    src_pattern = os.path.join(src_path, pattern)

    ffmpeg_exe = r"C:\Users\szizo\Desktop\ffmpeg\ffmpeg-2025-12-24-git-abb1524138-essentials_build\bin\ffmpeg.exe"

    # Build filtergraph
    vf_parts = []

    if resolution is not None:
        vf_parts.append(
            f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',"
            f"pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"
        )

    # Do FPS conversion *in the filtergraph* so we can see which frames were selected
    if fps is not None:
        vf_parts.append(f"fps={fps}")

    # Print per-frame info (includes n: frame index)
    vf_parts.append("showinfo")

    command = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel", "info",     # "verbose" if you want more
        "-sws_flags", mode,

        # image sequence timing (must be before -i)
        "-framerate", str(input_fps),
        "-i", src_pattern,

        "-an",
        "-threads", "10",
    ]

    if vf_parts:
        command += ["-vf", ",".join(vf_parts)]

    command += [dst_path]

    # Run and capture stderr (ffmpeg logs go to stderr)
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    # Save log if requested
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(result.stderr)

    # Optionally parse chosen frames from showinfo output
    if return_chosen_frames:
        # showinfo lines look like: ... showinfo ... n:123 pts:...
        chosen = []
        for line in result.stderr.splitlines():
            if "showinfo" in line and " n:" in line:
                m = re.search(r"\bn:(\d+)\b", line)
                if m:
                    chosen.append(int(m.group(1)))
        return chosen

    return None



# def ffmpeg_once_from_frames(
#     src_path: str,
#     dst_path: str,
#     *,
#     input_fps: int = 30,   # FPS the frame sequence represents (timeline)
#     fps: int = None,       # output FPS (e.g. 2)
#     resolution: int = None,
#     pad: str = "#000000",
#     mode: str = "bicubic",
#     pattern: str = "%05d.jpg",
# ):
#     os.makedirs(os.path.dirname(dst_path), exist_ok=True)

#     src_pattern = os.path.join(src_path, pattern)

#     command = [
#         r"C:\Users\szizo\Desktop\ffmpeg\ffmpeg-2025-12-24-git-abb1524138-essentials_build\bin\ffmpeg.exe",
#         "-y",
#         "-sws_flags", mode,

#         # image sequence timing (must be before -i)
#         "-framerate", str(input_fps),
#         "-i", src_pattern,

#         "-an",
#         "-threads", "10",
#     ]

#     if fps is not None:
#         command += ['-r', str(fps)]
#     if resolution is not None:
#         command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
#     vf.append("showinfo")
#     command += [dst_path]
#     subprocess.run(command, check=True)


def ffmpeg_once(src_path: str, dst_path: str, *, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    command = [
        './ffmpeg/ffmpeg',
        '-y',
        '-sws_flags', mode,
        '-i', src_path,
        '-an',
        '-threads', '10',
    ]
    if fps is not None:
        command += ['-r', str(fps)]
    if resolution is not None:
        command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
    command += [dst_path]
    # subprocess.run(command, check=True)


videosDir = r"C:\Users\szizo\Desktop\testCombine\merged"
outDir = r"C:\Users\szizo\Desktop\testCombine\merged_sampled_3"

os.makedirs(outDir, exist_ok=True)

logPath = os.path.join(outDir, "ffmpeg_showinfo.log")

for folderName in os.listdir(videosDir):

    folder = os.path.join(videosDir, folderName)

    if not os.path.isdir(folder):
        continue

    # embed_mark: str = '2fps_384_1+3x3'

    videoOutDir = os.path.join(outDir, f"{folderName}.mp4")

    ffmpeg_once_from_frames(src_path=folder, dst_path=videoOutDir, fps=2, resolution=384, log_path=logPath)
