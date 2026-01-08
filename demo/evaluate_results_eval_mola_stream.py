import json
import re
from tqdm import tqdm 

from typing import Dict

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


def get_isVideoViolent(videoPath):
    scenario =  int(re.findall(r"C\d+", videoPath)[0][1:])
    if scenario > 12:
        return False
    return True

def eval_results(results_list: list[dict]):
    correct_violent = 0
    gt_total_violent = 0

    correct_nonViolent = 0
    gt_total_nonViolent = 0
    for result in tqdm(results_list):
        videoName = result["videoName"]

        # isVideoViolent = result["isVideoViolent"]
        isVideoViolent = get_isVideoViolent(videoName) #remove Later

  


   
        found_violence_detected = False
        for obj in result["conversation"]:
            if "role" in obj and obj["role"] == "assistant" and obj["time"] != 0.0:
                if "Violence Detected" in obj["content"]:
                    found_violence_detected = True
                else:
                    print("For the followin Obj found assistant response not violence or query response!!")
                    print(obj)

        if isVideoViolent:
            gt_total_violent += 1
            if found_violence_detected:
                correct_violent += 1
        else:
            gt_total_nonViolent += 1
            if not found_violence_detected:
                correct_nonViolent += 1

    metrics = compute_metrics(correct_violent=correct_violent, gt_total_violent=gt_total_violent, correct_nonviolent=correct_nonViolent, gt_total_nonviolent=gt_total_nonViolent)
    print(metrics)





if __name__ == "__main__":
    anno_path = "/home/zeidan/Masters/videollm-online_fineTune_mola/demo/eval_mola_stream_results/test_results.jsonl"
    results_list = []
    with open(anno_path) as f:
        for line in f:
            results_list.append(json.loads(line))
    
    eval_results(results_list)

