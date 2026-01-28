from pathlib import Path
import json

root_dir = Path("/home/zeidan/Masters/videollm-online_fineTune_mola/demo/eval_mola_stream_results/train_38")


for anno in root_dir.rglob("*.jsonl"):
    if "0." in anno.name and "converted" in anno.name:
        print(anno.name)
        anno.unlink()