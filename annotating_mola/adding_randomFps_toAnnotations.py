import os, json, torch
import random
from pathlib import Path


annotations_dir_path = Path("/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withSampledNumFrames_segDescSummary")

out_annotations_dir_path = Path("/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withRandomFps")
out_annotations_dir_path.mkdir(parents=True, exist_ok=True)


# pt_root_path = Path("/home/zeidan/Desktop/preprocess/videos_sampled_1+3x3_google--siglip-large-patch16-384")

seed = 42
rng_fps = random.Random(seed)

fps_choices = [2, 3, 4]

for anno_file in annotations_dir_path.glob("*.json"):
    if "_total" in anno_file.name:
        print("skipping total for fps consistency.")
        continue
    updated_samples = []
    with open(anno_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
        print(f"Loaded annoFile: {anno_file.name}")
    for sample in samples:
        rand_fps = rng_fps.choice(fps_choices)
        sample["rand_fps"] = rand_fps
        updated_samples.append(sample)

    out_anno_file_path = out_annotations_dir_path / anno_file.name
    with open(out_anno_file_path, "w", encoding="utf-8") as out_f:
        json.dump(updated_samples, out_f, ensure_ascii=False, indent=4)
        print(f"Saved updated annoFile to: {out_anno_file_path}")