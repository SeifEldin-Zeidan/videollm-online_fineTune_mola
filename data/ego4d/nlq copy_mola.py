"""
Mola violence start detection dataset.

Expected annotation format (per split JSON):
[
  {
    "video_uid": "abc123",                  # basename of the encoded .pt file
    "starts": [12.5, 45.2],                 # list of violence start timestamps in seconds
    "duration": 120.0                       # optional, will fall back to encoded length
    # alternatively:
    # "violence": [{"start": 12.5}, {"start": 45.2}],
    # or "events": [{"start": 12.5, "label": "violence"}, ...]
  },
  ...
]

Videos must be pre-encoded to vision features at 2fps/384 (or matching args) and
saved under:
  {mola_root}/videos_{embed_mark}_{vision_pretrained.replace('/', '--')}/*.pt
"""

import os, json, torch, tqdm
from typing import List

from ..stream import StreamMixIn
from ..utils import DictWithTo, ceil_time_by_fps


class MolaViolenceStart(StreamMixIn):
    user_prompt = "Alert me as soon as violence begins in the video."
    detection_response = "I have detected violence."
    evaluation_kwargs = DictWithTo(evaluator='stream_evaluate')

    def __init__(
        self,
        *,
        split: str,
        frame_fps: int,
        vision_pretrained: str,
        embed_mark: str,
        system_prompt: str,
        tokenizer,
        is_training: bool,
        augmentation: bool = False,
        max_num_frames: int = 1200,
        mola_root: str = 'datasets/mola',
        anno_path: str | None = None,
        pre_event_seconds: float = 2.0,
        post_event_seconds: float = 2.0,
        **kwargs,
    ):
        assert split in ['train', 'val', 'test'], f"Unexpected split: {split}"
        self.split = split
        self.frame_fps = frame_fps
        self.vision_pretrained = vision_pretrained
        self.embed_mark = embed_mark
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds

        # paths
        self.root = mola_root
        self.video_root = os.path.join(self.root, 'videos')
        self.anno_root = os.path.join(self.root, 'annotations')
        self.embed_dir = f"{self.video_root}_{embed_mark}_{vision_pretrained.replace('/', '--')}"
        self.metadata = self._get_metadata()

        anno_path = anno_path or os.path.join(self.anno_root, f'violence_{split}.json')
        self.annos = self._build_annos(anno_path)

        super().__init__(
            is_training=is_training,
            system_prompt=system_prompt,
            augmentation=augmentation,
            max_num_frames=max_num_frames,
            tokenizer=tokenizer,
            **kwargs,
        )

    # ---------- helpers ----------
    def _get_metadata(self):
        metadata_path = f'{self.embed_dir}_metadata.json'
        if os.path.exists(metadata_path):
            return json.load(open(metadata_path))
        metadata = {}
        for file in tqdm.tqdm(os.listdir(self.embed_dir), desc=f'prepare {metadata_path}...'):
            path = os.path.join(self.embed_dir, file)
            duration = (len(torch.load(path)) - 1) / self.frame_fps
            key = os.path.splitext(os.path.basename(path))[0]
            metadata[key] = {'duration': duration, 'path': path}
        json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata

    @staticmethod
    def _extract_starts(anno: dict) -> List[float]:
        if 'starts' in anno:
            return anno['starts']
        if 'violence_starts' in anno:
            return anno['violence_starts']
        if 'violence' in anno and isinstance(anno['violence'], list):
            return [e['start'] for e in anno['violence'] if isinstance(e, dict) and 'start' in e]
        if 'events' in anno and isinstance(anno['events'], list):
            return [
                e['start'] for e in anno['events']
                if isinstance(e, dict) and 'start' in e and str(e.get('label', '')).lower() == 'violence'
            ]
        return []

    def _load_annotation_entries(self, anno_path: str) -> List[dict]:
        raw = json.load(open(anno_path))
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if 'videos' in raw and isinstance(raw['videos'], list):
                return raw['videos']
            if 'annotations' in raw and isinstance(raw['annotations'], list):
                return raw['annotations']
            # fall back: assume dict of video_uid -> payload
            return [dict(video_uid=k, **v) for k, v in raw.items()]
        raise ValueError(f'Unsupported annotation format in {anno_path}')

    def _build_annos(self, anno_path: str):
        entries = self._load_annotation_entries(anno_path)
        annos = []
        for entry in entries:
            if entry.get('split') and entry['split'] != self.split:
                continue
            video_uid = entry.get('video_uid') or entry.get('video_id') or entry.get('id')
            if not video_uid or video_uid not in self.metadata:
                continue
            duration = entry.get('duration', self.metadata[video_uid]['duration'])
            starts = self._extract_starts(entry)
            for start_time in starts:
                start_time = float(start_time)
                start_time = ceil_time_by_fps(start_time, self.frame_fps, 0, duration)
                window_start = max(0.0, start_time - self.pre_event_seconds)
                window_end = min(duration, start_time + self.post_event_seconds)
                start_frame = int(window_start * self.frame_fps)
                alert_frame = int(start_time * self.frame_fps)
                stop_frame = int(window_end * self.frame_fps) + 1
                num_frames_before_alert = max(1, alert_frame - start_frame + 1)

                conversation = [
                    {"role": "user", "content": self.user_prompt},
                    {"role": "stream", "num_frames": num_frames_before_alert, "learn": True},
                    {"role": "assistant", "content": self.detection_response, "learn": True},
                ]
                annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, stop_frame)},
                })
        return annos

    # ---------- dataset API ----------
    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = self.annos[index]
        return *super().__getitem__(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs


def build_mola_violence_start_train(**kwargs):
    return MolaViolenceStart(split='train', **kwargs)


def build_mola_violence_start_val(**kwargs):
    return MolaViolenceStart(split='val', **kwargs)


def build_mola_violence_start_test(**kwargs):
    return MolaViolenceStart(split='test', **kwargs)
