from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class LiveTrainingArguments(TrainingArguments):
    live_version: str = 'live1+'
    system_prompt: str = (
        "You are a vision-language model analyzing in-car surveillance video footage showing people seated "
        "in the backseat of a vehicle. Respond only when you detect an instance of violence in the streaming "
        "video, and respond with: 'Violence Detected!'. "
        "Do not respond if no violence is present, unless the user explicitly asks a question."
    )
    train_datasets: list[str] = None
    eval_datasets: list[str] = None
    videos_pt_root_dir: str | None = None #Mine
    annotations_root_dir: str | None = None #Mine
    stream_loss_weight: float = 1.0
    llm_pretrained: str = 'meta-llama/Meta-Llama-3-8B-Instruct'
    vision_pretrained: str = 'google/siglip-large-patch16-384'
    lora_modules: str = "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ['connector'])
    frame_fps: int = 2 # for training. inference can be 10
    frame_token_cls: bool = None
    frame_token_pooled: list[int] = None
    frame_resolution: int = 384
    frame_token_interval: str  = None
    frame_token_interval_threshold: float = 0.0
    augmentation: bool = False
    attn_implementation: str = 'flash_attention_2'
    output_dir: str = 'outputs/finetuning_debug'

@dataclass
class LiveOneTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1'
    frame_token_cls: bool = True
    frame_num_tokens: int = 1
    frame_token_interval: str  = ''
    embed_mark: str = '2fps_384_1'
    max_num_frames: int = 7200 # 1h, 2fps, 7200 frames

@dataclass
class LiveOnePlusTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1+'
    frame_token_cls: bool = True
    frame_token_pooled: list[int] = field(default_factory=lambda: [3,3])
    frame_num_tokens: int = 10 # 1+3x3
    embed_mark: str = '2fps_384_1+3x3'
    frame_token_interval: str = ','
    max_num_frames: int = 1200 # 10min, 2fps, 1200 frames

def get_args_class(live_version: str):
    if live_version == 'live1':
        return LiveOneTrainingArguments
    elif live_version == 'live1+':
        return LiveOnePlusTrainingArguments
    raise NotImplementedError
