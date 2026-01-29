from collections import defaultdict
import json
import math
import os
import re
from tqdm import tqdm 

from typing import Dict

from statistics import mean


from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt



def getPerScenarioAccuracy(perScenarioResults):
    perScenario_accuracy = {}
    for n in range(1, 21):
        scenario_gt = perScenarioResults[f"{n}_gt"]
        scenario_correct = perScenarioResults[f"{n}_correct"]
        scenario_acc = scenario_correct / scenario_gt if scenario_gt else -1
        perScenario_accuracy[f"C{n}"] = scenario_acc
    return perScenario_accuracy


def plot_accuracies_by_checkpoint(
    records: List[Dict[str, Any]],
    out_dir: Union[str, Path],
    *,
    checkpoint_key: str = "chkPnt",
    metric_prefix: str = "accuracy_",
    filename: str = "eval_set_accuracies_by_checkpoint.png",
    title: Optional[str] = "Accuracies vs Checkpoint",
) -> Path:
    if not records:
        raise ValueError("records is empty")

    # Sort records by checkpoint (treat checkpoint as int if possible)
    def _chk_as_int(d: Dict[str, Any]) -> int:
        v = d.get(checkpoint_key)
        if v is None:
            raise KeyError(f"Missing '{checkpoint_key}' in record: {d}")
        try:
            return int(v)
        except (TypeError, ValueError):
            # fallback: string sort order
            return int(re.findall(r"\d+", str(v))[0]) if re.findall(r"\d+", str(v)) else 0

    records = sorted(records, key=_chk_as_int)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [_chk_as_int(r) for r in records]

    # metrics to plot
    metric_keys = sorted({k for r in records for k in r.keys() if k.startswith(metric_prefix)})
    if not metric_keys:
        raise ValueError(f"No metrics found with prefix '{metric_prefix}'")

    # Plot (matplotlib will auto-assign different colors per line)
    plt.figure()
    for k in metric_keys:
        ys = [r.get(k, None) for r in records]
        plt.plot(checkpoints, ys, marker="o", label=k)

    plt.xlabel("Checkpoint")
    plt.ylabel("Accuracy")
    if title:
        plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    out_path = out_dir / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path

def compute_metrics(
    correct_violent: int,
    gt_total_violent: int,
    correct_nonviolent: int,
    gt_total_nonviolent: int,
) -> Dict[str, float]:
    """
    Compute binary classification metrics for violence detection.

    Returns a dict with:
    - per-class accuracy
    - overall accuracy
    - precision / recall / F1 for violent class
    - false positive / false negative rates
    """

    # Confusion matrix terms
    TP = correct_violent
    TN = correct_nonviolent
    FN = gt_total_violent - correct_violent
    FP = gt_total_nonviolent - correct_nonviolent

    total = gt_total_violent + gt_total_nonviolent

    # Accuracies
    acc_violent = TP / gt_total_violent if gt_total_violent > 0 else 0.0
    acc_nonviolent = TN / gt_total_nonviolent if gt_total_nonviolent > 0 else 0.0
    acc_overall = (TP + TN) / total if total > 0 else 0.0

    # Precision / Recall / F1 (violent class)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / gt_total_violent if gt_total_violent > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Error rates
    false_positive_rate = FP / gt_total_nonviolent if gt_total_nonviolent > 0 else 0.0
    false_negative_rate = FN / gt_total_violent if gt_total_violent > 0 else 0.0

    return {
        "accuracy_overall": acc_overall,
        "accuracy_violent": acc_violent,
        "accuracy_nonviolent": acc_nonviolent,
        "count_violence": gt_total_violent,
        "count_nonViolence": gt_total_nonviolent,
        # "precision_violent": precision,
        # "recall_violent": recall,
        # "f1_violent": f1,
        # "false_positive_rate": false_positive_rate,
        # "false_negative_rate": false_negative_rate,
        # "tp": TP,
        # "tn": TN,
        # "fp": FP,
        # "fn": FN,
    }

def get_scenario_num(videoPath):
    scenario =  int(re.findall(r"C\d+", videoPath)[0][1:])
    return scenario

def get_isVideoViolent(videoPath):
    scenario =  int(re.findall(r"C\d+", videoPath)[0][1:])
    if scenario > 12:
        return False
    return True

time_diff_gt_detection_after = []
time_diff_gt_detection_before = []
time_diff_gt_delayed_detection_after = []
time_diff_gt_delayed_detection_before = []


def print_time_diff_metric():

    


    avg_detection_after = mean(time_diff_gt_detection_after)
    avg_detection_before = mean(time_diff_gt_detection_before)
    avg_delayed_after = mean(time_diff_gt_delayed_detection_after)
    avg_delayed_before = mean(time_diff_gt_delayed_detection_before)

    after_perc = len(time_diff_gt_detection_after) / (len(time_diff_gt_detection_after) +  len(time_diff_gt_detection_before))
    before_perc = len(time_diff_gt_detection_before) / (len(time_diff_gt_detection_after) +  len(time_diff_gt_detection_before))

    delayed_after_perc = len(time_diff_gt_delayed_detection_after) / (len(time_diff_gt_delayed_detection_after) +  len(time_diff_gt_delayed_detection_before))
    delayed_before_perc = len(time_diff_gt_delayed_detection_before) / (len(time_diff_gt_delayed_detection_after) +  len(time_diff_gt_delayed_detection_before))

    results = {
        "avg_detection_after": avg_detection_after,
        "avg_detection_before": avg_detection_before,
        "perc_after": after_perc,
        "perc_before": before_perc,



        "avg_delayed_after": avg_delayed_after,
        "avg_delayed_before": avg_delayed_before,
        "perc_delayed_after": delayed_after_perc,
        "perc_delayed_before": delayed_before_perc,
    }
    print("\nFor gt onset exact:")
    print(f"\nDetections Delay avg = {avg_detection_after} secs")
    print(f"Detections Early avg = {avg_detection_before} secs")
    print(f"Detections Delay perc = {after_perc * 100} %")
    print(f"Detections Early perc = {before_perc * 100} %")

    print("\nFor gt delayed (like training) onset:")
    print(f"\nDetections Delay avg = {avg_delayed_after} secs")
    print(f"Detections Early avg = {avg_delayed_before} secs")
    print(f"Detections Delay perc = {delayed_after_perc * 100} %")
    print(f"Detections Early perc = {delayed_before_perc * 100} %")

    return results



def time_diff_metric(videoName, detected_time):
    global time_diff_gt_detection_after, time_diff_gt_detection_before, time_diff_gt_delayed_detection_after, time_diff_gt_delayed_detection_before
    sample_anno = dataset_annos[videoName]
    gt_response_time, delayed_gt_response_time = sample_anno["gt_response_time"], sample_anno["delayed_gt_response_time"]

   

    time_diff_gt = gt_response_time - detected_time

    if time_diff_gt < 0:
        time_diff_gt_detection_after.append(time_diff_gt * -1)
    else:
        time_diff_gt_detection_before.append(time_diff_gt)


  
    time_diff_gt_delayed = delayed_gt_response_time - detected_time

    if time_diff_gt_delayed < 0:
        time_diff_gt_delayed_detection_after.append(time_diff_gt_delayed * -1)
    else:
        time_diff_gt_delayed_detection_before.append(time_diff_gt_delayed)



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

def eval_results(results_list: list[dict]):
    perScenarioResults = defaultdict(int)
    correct_violent = 0
    gt_total_violent = 0

    correct_nonViolent = 0
    gt_total_nonViolent = 0
    for result in tqdm(results_list):
        videoName = result["videoName"]

        isVideoViolent = result["isVideoViolent"]
        # isVideoViolent = get_isVideoViolent(videoName) #remove Later

  


   
        found_violence_detected = False
        detected_time = None
        for obj in result["conversation"]:
            if "role" in obj and obj["role"] == "assistant" and obj["time"] != 0.0:
                if "Assistant: Violence Detected!" in obj["content"]:
                    if found_violence_detected:
                        print("For the followin Obj found MULTIPLE assistant response with violence detected!!")
                        print(obj)
                    found_violence_detected = True
                    detected_time = obj["time"]
                else:
                    print("For the followin Obj found assistant response not violence or query response!!")
                    print(obj)

        scenario_num = get_scenario_num(videoName)
        perScenarioResults[f"{scenario_num}_gt"] += 1
        if isVideoViolent:
            gt_total_violent += 1
            if found_violence_detected:
                correct_violent += 1
                perScenarioResults[f"{scenario_num}_correct"] += 1
                time_diff_metric(videoName, detected_time)
        else:
            gt_total_nonViolent += 1
            if not found_violence_detected:
                correct_nonViolent += 1
                perScenarioResults[f"{scenario_num}_correct"] += 1

    metrics = compute_metrics(correct_violent=correct_violent, gt_total_violent=gt_total_violent, correct_nonviolent=correct_nonViolent, gt_total_nonviolent=gt_total_nonViolent)
    print(metrics)
    try:
        results = print_time_diff_metric()
        metrics = {**metrics, **results} 
    except:
        print("Couldnt get time diff metric!")

    perScenario_acc = getPerScenarioAccuracy(perScenarioResults)
    metrics = {**metrics, **perScenario_acc} 

    return metrics
    





if __name__ == "__main__":
    bestAcc = 0
    best_chk = None
    best_chk_metrics = None
    anno_path_root = "/home/zeidan/Masters/videollm-online_fineTune_mola/demo/eval_mola_stream_results/train_32LearnB/chk_112"
    val = True  #True for val, False for test
    count_total = 317
    count_lower_than_total = []
    metrics_chkPnt = []
    for root,dirs,files in os.walk(anno_path_root):
        # files = [f for f in files if f.endswith(".jsonl")]
        # files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
        for filename in files:
            if filename.endswith(".jsonl"):
                # if filename != "test_results_2fps.json.jsonl":
                #     continue

                if "0." in filename:
                    continue

                if "converted" in filename:
                    continue
                if val and "val" not in filename:
                    continue
                elif not val and "test" not in filename:
                    continue
                anno_path = os.path.join(root, filename)
                chkPnt = os.path.basename(os.path.dirname(anno_path)).split("_")[-1]
                print("=" *10)
                print(f"Processing checkPoint: {chkPnt}")
                print(f"testing results in: {filename}")
    
                results_list = []
                with open(anno_path) as f:
                    for line in f:
                        results_list.append(json.loads(line))
                
                frame_fps_test = 2
                max_shiftViolenceStart_time = 3
                if val:
                    anno_path_testSet = "/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withRandomFps_withSampledNumFrames/vLLMonline_mola_val.json"
                else:
                    anno_path_testSet = "/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withRandomFps_withSampledNumFrames/vLLMonline_mola_test.json"
                dataset_annos = loadAnnotations(anno_path=anno_path_testSet, frame_fps = frame_fps_test, max_shiftViolenceStart_time = max_shiftViolenceStart_time)
                metrics = eval_results(results_list)
                total_ran = metrics["count_violence"] + metrics["count_nonViolence"]
                if total_ran < count_total:
                    count_lower_than_total.append((chkPnt, metrics))
                    print(f"Skipped chk pnt: {chkPnt} has {total_ran} samples which is less then count_total = {count_total}")
                    continue #skip rest, not appended in graph

                if metrics["accuracy_overall"] > bestAcc:
                    bestAcc = metrics["accuracy_overall"]
                    best_chk = chkPnt
                    best_chk_metrics = metrics

                metrics["chkPnt"] = chkPnt
                metrics_chkPnt.append(metrics)
                time_diff_gt_detection_after.clear()
                time_diff_gt_detection_before.clear()
                time_diff_gt_delayed_detection_after.clear()
                time_diff_gt_delayed_detection_before.clear()

    # print(metrics_chkPnt)
    if len(metrics_chkPnt) > 1 and val:
        plot_accuracies_by_checkpoint(metrics_chkPnt, anno_path_root, filename="eval_set_accuracies_by_checkpoint.png")
    print("\n" * 2)
    print("count_lower_than_total:")
    print(len(count_lower_than_total))
    print(count_lower_than_total)
        
    
    print("=" * 10)
    print(f"best overall acc = {bestAcc}")
    print(f"For Checkpoint = {best_chk}")
    print(best_chk_metrics)