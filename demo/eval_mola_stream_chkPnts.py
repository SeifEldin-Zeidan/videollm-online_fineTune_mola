
import argparse
import math
import os, torchvision, transformers, time, json
import random
import sys
# from tqdm import tqdm
import tqdm
import subprocess
import torch.multiprocessing as mp

from data.utils import ffmpeg_once

from .inference_eval import LiveInfer

from collections import defaultdict
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 
# --resume_from_checkpoint /netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/outputs/mola_live1+/train_14/


seed = 42
flipRng = random.Random(seed)


def main(liveinfer: LiveInfer, dataset_annos, pt_root_path, results_path):


    results_json_list = []
    for videoName, videoAnno in tqdm.tqdm(dataset_annos.items()):
        print(f"Processing video: {videoName}")
        # sample_video_path = os.path.join(videos_root_path, videoName)
        sample_pt_path = os.path.join(pt_root_path, f"{videoName}.pt")

        flip = flipRng.choice([True, False])
        if flip:
            sample_pt_path = sample_pt_path.replace("1+3x3", "flipped_1+3x3")



        isVideoViolent = videoAnno["isViolent"]

        liveinfer.load_video(sample_pt_path)
        

        liveinfer.input_query_stream(violence_query, video_time=0)

        num_video_frames = liveinfer.num_video_frames


        timecosts = []
        pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
        history = {'videoName': videoName, 'isVideoViolent': isVideoViolent, 'frame_fps': liveinfer.frame_fps, 'conversation': []} 
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
            if query:
                history['conversation'].append({'role': 'user', 'content': query, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
                print(query)
            if response:
                history['conversation'].append({'role': 'assistant', 'content': response, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
                print(response)
            if not query and not response:
                history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
        # json.dump(history, open(save_history_path, 'a'), indent=4)
        results_json_list.append(history)
        liveinfer.reset()
        with open(results_path, "a", encoding="utf-8") as results_jsonlFile:
            json.dump(history, results_jsonlFile)
            results_jsonlFile.write("\n")
    print("Finished Evaluation!")
    print(f"Saved annotations to {results_path}")
    return results_json_list


def convertNumFrames_toNewFps(numFrames, originalFps, newFps):
    new_numFrames = math.ceil((numFrames/originalFps) * newFps)
    return new_numFrames

def get_shiftViolenceStart(numFrames_restOfVideo, frame_fps, max_shiftViolenceStart_time):
        # shift_perc_v = math.ceil(numFrames_restOfVideo * self.shiftViolenceStart_perc)
        max_numFrames_shift_bySec = math.ceil(max_shiftViolenceStart_time * frame_fps)
        return min(numFrames_restOfVideo-frame_fps, max_numFrames_shift_bySec)

def loadAnnotations(anno_path, frame_fps, max_shiftViolenceStart_time):
    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    annotations_dict = defaultdict(dict)
    for anno in annotations:
        videoViolent = anno["numFrames_violent_segment"] > 0
        if videoViolent:
            numFrames_nonViolent_segment_converted = convertNumFrames_toNewFps(
                numFrames=anno["numFrames_nonviolent_segment"],
                originalFps=30,
                newFps=frame_fps
                )
            gt_response_time = (numFrames_nonViolent_segment_converted + 1) / frame_fps

            training_delay = get_shiftViolenceStart(
                numFrames_restOfVideo = anno["numFrames_sampled"] - numFrames_nonViolent_segment_converted,
                frame_fps=frame_fps,
                max_shiftViolenceStart_time=max_shiftViolenceStart_time
            )

            delayed_gt_response_time = (numFrames_nonViolent_segment_converted + training_delay) / frame_fps

            annotations_dict[anno["videoName"]] = {
                "isViolent": videoViolent,
                "gt_response_time": gt_response_time,
                "delayed_gt_response_time": delayed_gt_response_time
            }
        else:
            annotations_dict[anno["videoName"]] = {
                "isViolent": videoViolent,
                "gt_response_time": 0,
                "delayed_gt_response_time": 0
            }
    return annotations_dict

# def evaluateDetectionAccuracy(dataset_annos, results_json_list):
#     correct_violent = []
#     gt_total_violent = []

#     correct_nonViolent = []
#     gt_total_nonViolent = []

#     for res_json in results_json_list:
#         if res

if __name__ == '__main__': #run with 2 fps annotations


    

    anno_path_def = "/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withSampledNumFrames_segDescSummary/vLLMonline_mola_test.json"

    pt_root_path_def = "/netscratch/zeidan/mola_segments_combined_ftVLLMOnline/videos_sampled_1+3x3_google--siglip-large-patch16-384"
    results_root_path_def = "/netscratch/zeidan/finetune_videoLLM_online/videollm-online_fineTune_mola/demo/eval_mola_stream_results"
    
    violence_query_def = "Analyze the given surveilance video and respond only when you detect an instance of violence in the streaming video, and respond with: 'Violence Detected!'. Do not respond if no violence is present."
    user_summary_query_def = "Based on the preceding video frames, determine whether any violent behavior is present. Respond in exactly the following structure describing what you have seen. Violence Detected: [Yes/No]\nDescription: [Description of the actions in the video]\nAttacker: [Whether the left or right passenger is doing the violence, if any; otherwise 'None']\nCategory: [Interaction type]."
    frame_fps_def = 2
    max_shiftViolenceStart_time_def = 1.5

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_num", type=str, required=True)
    parser.add_argument("--chk_num", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=False, default=anno_path_def)
    parser.add_argument("--pt_root_path", type=str, required=False, default=pt_root_path_def)
    parser.add_argument("--results_root_path", type=str, required=False, default=results_root_path_def)
    parser.add_argument("--results_fileName", type=str, required=False, default=None)
    parser.add_argument("--violence_query", type=str, required=False, default=violence_query_def)
    parser.add_argument("--user_summary_query", type=str, required=False, default=user_summary_query_def)
    parser.add_argument("--frame_fps", type=int, required=False, default=frame_fps_def)
    parser.add_argument("--max_shiftViolenceStart_time", type=int, required=False, default=max_shiftViolenceStart_time_def)

    # args = parser.parse_args()
    script_args, model_args = parser.parse_known_args()

    anno_path = script_args.anno_path
    pt_root_path = script_args.pt_root_path
    results_root_path = script_args.results_root_path
    results_fileName = script_args.results_fileName
    violence_query = script_args.violence_query
    user_summary_query = script_args.user_summary_query
    frame_fps = script_args.frame_fps
    max_shiftViolenceStart_time = script_args.max_shiftViolenceStart_time
    chk_num = script_args.chk_num
    train_num = script_args.train_num

    results_root_path = os.path.join(results_root_path, f"train_{train_num}", f"chk_{chk_num}")

    os.makedirs(results_root_path, exist_ok=True)

    if not results_fileName:
        annotype = anno_path_def.split("_")[-1].split(".")[0]
        results_path = os.path.join(results_root_path, f"{annotype}_results.jsonl")
    else:
        results_path = os.path.join(results_root_path, results_fileName)

    old_argv = sys.argv
    sys.argv = [old_argv[0]] + model_args

    print(f"old_argv = {old_argv}")
    print(f"model_args = {model_args}")

    liveinfer = LiveInfer()
    liveinfer.violence_query = violence_query 

    liveinfer.frame_fps = frame_fps
    
    sys.argv = old_argv

    dataset_annos = loadAnnotations(anno_path, frame_fps, max_shiftViolenceStart_time)
    
    results_json_list = main(liveinfer, dataset_annos, pt_root_path=pt_root_path, results_path=results_path)
