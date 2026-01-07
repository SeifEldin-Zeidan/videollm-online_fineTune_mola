import os, json, torch
from pathlib import Path


annotations_dir_path = Path("/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withRandomFps")

out_annotations_dir_path = Path("/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withRandomFps_withSampledNumFrames")
out_annotations_dir_path.mkdir(parents=True, exist_ok=True)


pt_root_path = Path("/home/zeidan/Desktop/preprocess/videos_sampled_randFps_1+3x3_google--siglip-large-patch16-384")


for anno_file in annotations_dir_path.glob("*.json"):
    updated_samples = []
    with open(anno_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
        print(f"Loaded annoFile: {anno_file.name}")
    for sample in samples:
        sample_name = sample["videoName"]
        pt_file_path = pt_root_path / f"{sample_name}.pt"

        video_pt = torch.load(pt_file_path, map_location="cpu")
        video_numFrames = tuple(video_pt.shape)[0]

        sample["numFrames_sampled"] = video_numFrames
        updated_samples.append(sample)

    out_anno_file_path = out_annotations_dir_path / anno_file.name
    with open(out_anno_file_path, "w", encoding="utf-8") as out_f:
        json.dump(updated_samples, out_f, ensure_ascii=False, indent=4)
        print(f"Saved updated annoFile to: {out_anno_file_path}")