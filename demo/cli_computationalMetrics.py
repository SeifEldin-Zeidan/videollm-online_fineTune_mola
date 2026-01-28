import torch
import os, torchvision, transformers, tqdm, time, json
import subprocess
import torch.multiprocessing as mp

from data.utils import ffmpeg_once

from .inference import LiveInfer
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

def main(liveinfer: LiveInfer):
    # src_video_path ='demo/assets/C1_P1_P2_1_Se1.mp4'
    # src_video_path ='demo/assets/C2_P14_P15_1_Se1.mp4'
    src_video_path = 'demo/assets/C1_P16_P15_2_Se2.mp4'
    # src_video_path = 'demo/assets/C5_P13_P14_2_Se1.mp4'
    # src_video_path ='demo/assets/C16_P14_P15_1_Se1.mp4'
    # src_video_path = 'demo/assets/C19_P15_P16_1_Se2.mp4'
    # src_video_path = 'demo/assets/C19_P14_P15_1_Se1.mp4'
    # src_video_path = 'demo/assets/C20_P16_P15_2_Se2.mp4'

    name, ext = os.path.splitext(src_video_path)
    ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
    # ffmpeg_video_path = os.path.join('demo/assets/cache', name + f'_{liveinfer.frame_fps}fps_{None}' + ext)
    save_history_path = src_video_path.replace('.mp4', '.json')
    if not os.path.exists(ffmpeg_video_path):
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
        # ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=None)
        logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')



    duplicate_numTimes = 12
    new_ffmpeg_video_path = ffmpeg_video_path.replace(ext,f"_dup_{duplicate_numTimes}{ext}")
    if not os.path.exists(new_ffmpeg_video_path):
        duplicate_video_multipleTimes(ffmpeg_video_path, new_ffmpeg_video_path, duplicate_numTimes = duplicate_numTimes)
        print(f"new duplicate video path is:  {new_ffmpeg_video_path}")
    ffmpeg_video_path = new_ffmpeg_video_path


    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before_video = torch.cuda.memory_allocated()

    liveinfer.load_video(ffmpeg_video_path)
    torch.cuda.synchronize()

    mem_after_video = torch.cuda.memory_allocated()
    video_mem_bytes = mem_after_video - mem_before_video
    video_mem_mb = video_mem_bytes / 1024**2

    print("\n")
    print("=" * 8)
    print(f"Video tensor VRAM: {video_mem_mb:.1f} MB")
    print("=" * 8)
    print("\n" )


    num_video_frames = liveinfer.num_video_frames

    print(f"num_video_frames = {num_video_frames}!!!")
    print(f"liveinfer.frame_fps = {liveinfer.frame_fps}")
    print(f"liveinfer.video_duration = {liveinfer.video_duration}")

    user_summary_query = "Based on the preceding video frames, determine whether any violent behavior is present. Respond in exactly the following structure describing what you have seen. Violence Detected: [Yes/No]\nDescription: [Description of the actions in the video]\nAttacker: [Whether the left or right passenger is doing the violence, if any; otherwise 'None']\nCategory: [Interaction type]."
    violence_query = "Analyze the given surveilance video and respond only when you detect an instance of violence in the streaming video, and respond with: 'Violence Detected!'. Do not respond if no violence is present."
    liveinfer.violence_query = violence_query

    liveinfer.input_query_stream(violence_query, video_time=0)
    # liveinfer.input_query_stream(user_summary_query, video_time=liveinfer.video_duration - 0.5)

    # liveinfer.input_query_stream('Please narrate the video in real time.', video_time=0.0)
    # liveinfer.input_query_stream('Respond as soon as you detect a violence instance in the video', video_time=0.0)
    # liveinfer.input_query_stream('Hi, who are you?', video_time=1.0)
    # liveinfer.input_query_stream('Yes, I want to check its safety.', video_time=3.0)
    # liveinfer.input_query_stream('No, I am going to install something to alert pedestrians to move aside. Could you guess what it is?', video_time=12.5)



    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    baseline_alloc = torch.cuda.memory_allocated() / 1024**2
    baseline_reserved = torch.cuda.memory_reserved() / 1024**2

    print(f"Baseline allocated VRAM: {baseline_alloc:.1f} MB")
    print(f"Baseline reserved VRAM:  {baseline_reserved:.1f} MB")


    perf_start_time = time.perf_counter()


    timecosts = []
    pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
    history = {'video_path': src_video_path, 'frame_fps': liveinfer.frame_fps, 'conversation': []} 
    for i in range(num_video_frames):
        # liveinfer.frame_token_interval_threshold -= 0.00175 # decay
        start_time = time.time()
        liveinfer.input_video_stream(i / liveinfer.frame_fps)
        query, response = liveinfer()
        end_time = time.time()
        timecosts.append(end_time - start_time)
        fps = (i + 1) / sum(timecosts)
        pbar.set_postfix_str(f"Average Processing FPS: {fps:.1f}")
        pbar.update(1)
        # if query:
        #     history['conversation'].append({'role': 'user', 'content': query, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
        #     print(query)
        # if response:
        #     history['conversation'].append({'role': 'assistant', 'content': response, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
        #     print(response)
        # if not query and not response:
        #     history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
    
    perf_end_time = time.perf_counter()
    perf_elapsed = perf_end_time - perf_start_time

    print("\n" * 2)
    print("=" * 8)
    print(f"perf_elapsed_time = {perf_elapsed}")
    print("=" * 8)
    print("\n" * 2)


    after_alloc = torch.cuda.memory_allocated() / 1024**2
    after_reserved = torch.cuda.memory_reserved() / 1024**2
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

    print("\n" * 2)
    print("=" * 8)

    print(f"After allocated VRAM: {after_alloc:.1f} MB")
    print(f"After reserved VRAM:  {after_reserved:.1f} MB")
    print(f"Peak allocated VRAM:  {peak_alloc:.1f} MB")
    print(f"Peak reserved VRAM:   {peak_reserved:.1f} MB")
    
    print("=" * 8)
    print("\n" * 2)



    json.dump(history, open(save_history_path, 'a'), indent=4)
    print(f'The conversation history has been saved to {save_history_path}.')


def duplicate_video(input_path, output_path):
    """
    Duplicates a video by concatenating it with itself once.
    Example: 15s -> 30s
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create a temporary file list for ffmpeg concat
    list_file = "concat_list.txt"
    with open(list_file, "w") as f:
        f.write(f"file '{os.path.abspath(input_path)}'\n")
        f.write(f"file '{os.path.abspath(input_path)}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ]

    subprocess.run(cmd, check=True)
    os.remove(list_file)

def duplicate_video_multipleTimes(input_path, output_path, duplicate_numTimes):
    """
    Duplicates a video by concatenating it with itself once.
    Example: 15s -> 30s
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create a temporary file list for ffmpeg concat
    list_file = "concat_list.txt"
    with open(list_file, "w") as f:
        for i in range(duplicate_numTimes):
            f.write(f"file '{os.path.abspath(input_path)}'\n")
        # f.write(f"file '{os.path.abspath(input_path)}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path
    ]

    subprocess.run(cmd, check=True)
    os.remove(list_file)

if __name__ == '__main__':
    liveinfer = LiveInfer()
    main(liveinfer)