from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess

import re

from concurrent.futures import ProcessPoolExecutor, as_completed

from collections import defaultdict

# def ffmpeg_once_from_frames(
#     src_path: str,
#     dst_path: str,
#     fps: int,       # output FPS (e.g. 2)
#     resolution: int,
#     *,
#     input_fps: int = 30,   # FPS the frame sequence represents (timeline)
#     pad: str = "#000000",
#     mode: str = "bicubic",
#     pattern: str = "%05d.jpg",
#     log_path: str = None,  # if set, saves ffmpeg output log here
#     return_chosen_frames: bool = True,
# ):
#     os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

#     src_pattern = os.path.join(src_path, pattern)

#     ffmpeg_exe = r"C:\Users\szizo\Desktop\ffmpeg\ffmpeg-2025-12-24-git-abb1524138-essentials_build\bin\ffmpeg.exe"

#     # Build filtergraph
#     vf_parts = []

#     if resolution is not None:
#         vf_parts.append(
#             f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',"
#             f"pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"
#         )

#     # Do FPS conversion *in the filtergraph* so we can see which frames were selected
#     if fps is not None:
#         vf_parts.append(f"fps={fps}")

#     # Print per-frame info (includes n: frame index)
#     vf_parts.append("showinfo")

#     command = [
#         ffmpeg_exe,
#         "-y",
#         "-hide_banner",
#         "-loglevel", "info",     # "verbose" if you want more
#         "-sws_flags", mode,

#         # image sequence timing (must be before -i)
#         "-framerate", str(input_fps),
#         "-i", src_pattern,

#         "-an",
#         "-threads", "10",
#     ]

#     if vf_parts:
#         command += ["-vf", ",".join(vf_parts)]

#     command += [dst_path]

#     # Run and capture stderr (ffmpeg logs go to stderr)
#     result = subprocess.run(command, check=True, capture_output=True, text=True)

#     # Save log if requested
#     if log_path is not None:
#         os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
#         with open(log_path, "w", encoding="utf-8") as f:
#             f.write(result.stderr)

#     # Optionally parse chosen frames from showinfo output
#     if return_chosen_frames:
#         # showinfo lines look like: ... showinfo ... n:123 pts:...
#         chosen = []
#         for line in result.stderr.splitlines():
#             if "showinfo" in line and " n:" in line:
#                 m = re.search(r"\bn:(\d+)\b", line)
#                 if m:
#                     chosen.append(int(m.group(1)))
#         return chosen

#     return None



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


def ffmpeg_once_frames(src_path: str, dst_path: str, fps: int, resolution: int, *, pad: str = '#000000', mode='bicubic', input_fps=30):
    # os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Use image sequence instead of mp4
    src_pattern = os.path.join(src_path, "%05d.jpg")

    # ffmpeg_exe = r"C:\Users\szizo\Desktop\ffmpeg\ffmpeg-2025-12-24-git-abb1524138-essentials_build\bin\ffmpeg.exe"
    ffmpeg_path = "/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/data/preprocess/ffmpeg/ffmpeg"

    command = [
        # './ffmpeg/ffmpeg',
        ffmpeg_path,
        '-y',
        '-sws_flags', mode,
        "-framerate", str(input_fps),
        '-i', src_pattern,
        '-an',
        '-threads', '2',
    ]
    if fps is not None:
        command += ['-r', str(fps)]
    if resolution is not None:
        command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
    command += [dst_path]

    subprocess.run(command, check=True)


def ffmpeg_once(src_path: str, dst_path: str, *, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    command = [
        './ffmpeg/ffmpeg',
        '-y',
        '-sws_flags', mode,
        '-i', src_path,
        '-an',
        '-threads', '2',
    ]
    if fps is not None:
        command += ['-r', str(fps)]
    if resolution is not None:
        command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
    command += [dst_path]
    # subprocess.run(command, check=True)



def process_one(folderName: str, videosDir: str, outDir: str):
    folder = os.path.join(videosDir, folderName)
    if not os.path.isdir(folder):
        return None

    videoOutPath = os.path.join(outDir, f"{folderName}.mp4")
    print(f"\n!!!!!!!!!Sampling video {folderName}!!!!!!!!\n")

    randFps = fps_dict[folderName]
    print(f"Rand Fps = {randFps}")

    ffmpeg_once_frames(src_path=folder, dst_path=videoOutPath, fps=randFps, resolution=384)
    return folderName

def load_fps(annotatations_dir_path):
    global fps_dict
    annotatations_dir_path = Path(annotatations_dir_path)
    for anno_file in annotatations_dir_path.glob("*.json"):
        with open(anno_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
            print(f"Loaded annoFile: {anno_file.name}")
        for sample in samples:
            videoName = sample["videoName"]
            fps_dict[videoName] = sample["rand_fps"]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--input_frames_dir", type=str, default="/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/videos_train_val_test_split", required=False)
    parser.add_argument("--out_videos_dir", type=str, default="/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/videos_sampled_rand", required=False)

    args = parser.parse_args()

    # videosDir = r"C:\Users\szizo\Desktop\testCombine\merged"
    # outDir = r"C:\Users\szizo\Desktop\testCombine\merged_sampled_final"

    videosDir = args.input_frames_dir
    outDir = args.out_videos_dir

    os.makedirs(outDir, exist_ok=True)

    # logPath = os.path.join(outDir, "ffmpeg_showinfo.log")



    # for folderName in os.listdir(videosDir):

    #     folder = os.path.join(videosDir, folderName)

    #     if not os.path.isdir(folder):
    #         continue

    #     print(f"\n!!!!!!!!!Sampling video {folderName}!!!!!!!!\n")

    #     # embed_mark: str = '2fps_384_1+3x3'

    #     videoOutDir = os.path.join(outDir, f"{folderName}.mp4")

    #     ffmpeg_once_frames(src_path=folder, dst_path=videoOutDir, fps=2, resolution=384)


    fps_dict = defaultdict(int)
    annotatations_dir_path = "/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withRandomFps"
    load_fps(annotatations_dir_path)




    folder_names = [n for n in os.listdir(videosDir) if os.path.isdir(os.path.join(videosDir, n))]

    workers = min(8, (os.cpu_count() or 8))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one, name, videosDir, outDir) for name in folder_names]

        for f in as_completed(futures):
            try:
                done = f.result()
                if done is not None:
                    print(f"Done: {done}")
            except Exception as e:
                print(f"FAILED: {e}")
