import json, os
from collections import defaultdict
import statistics

from collections import deque

import numpy as np
# from pathlib import Path

def save_hparam_heatmap(
    results,
    hp1: str,
    hp2: str,
    metric: str,
    file_name: str = "heatmap.png",
    out_root_path: str = "/home/zeidan/Masters/videollm-online_fineTune_mola/demo/analyze_analyzeProbResults_saved_graphs",
    title: str | None = None,
    agg: str = "mean",
    annotate: bool = False,
    fmt: str = ".3f",
    dpi: int = 200,
):
    """
    Save a heatmap for a 2D hyperparameter sweep.

    Args:
        results: Iterable of dicts OR a pandas DataFrame with columns [hp1, hp2, metric].
                 Example row: {"lr": 1e-4, "wd": 1e-3, "f1": 0.83}
        hp1: Name of hyperparameter for Y-axis (rows).
        hp2: Name of hyperparameter for X-axis (cols).
        metric: Metric column name (heatmap values).
        out_path: Where to save (e.g., "grid_f1.png" or "grid_f1.pdf").
        title: Optional plot title.
        agg: If duplicate (hp1,hp2) pairs exist, how to aggregate: "mean", "max", "min", "median".
        annotate: If True, write values into cells.
        fmt: Format string for annotations.
        dpi: Output DPI for raster formats.

    Returns:
        (pivot_df, fig, ax) where pivot_df is the grid used for plotting.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize


    os.makedirs(out_root_path, exist_ok=True)

    out_path = os.path.join(out_root_path, file_name)
    

    # --- build DataFrame ---
    if hasattr(results, "columns"):  # pandas DataFrame-like
        df = results.copy()
    else:
        df = pd.DataFrame(results)

    for col in (hp1, hp2, metric):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Available: {list(df.columns)}")

    df = df[[hp1, hp2, metric]].dropna()

    # --- handle duplicates by aggregating ---
    if df.duplicated(subset=[hp1, hp2]).any():
        if agg not in {"mean", "max", "min", "median"}:
            raise ValueError("agg must be one of: 'mean', 'max', 'min', 'median'")
        df = (
            df.groupby([hp1, hp2], as_index=False)[metric]
            .agg(agg)
        )

    # --- pivot to grid ---
    pivot = df.pivot(index=hp1, columns=hp2, values=metric)

    # Sort axes if possible (numeric / sortable). If not, keep original order.
    try:
        pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    except Exception:
        pass

    values = pivot.to_numpy(dtype=float)

    # --- plot ---
    fig, ax = plt.subplots()

    # Robust scaling that ignores NaNs
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("No finite metric values to plot.")
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(values, aspect="auto", origin="lower", norm=norm)

    # ticks + labels
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))

    # Use readable labels (keep scientific notation for floats)
    xlabels = [str(c) for c in pivot.columns.tolist()]
    ylabels = [str(i) for i in pivot.index.tolist()]
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)

    ax.set_xlabel(hp2)
    ax.set_ylabel(hp1)
    ax.set_title(title or f"{metric} over ({hp1}, {hp2})")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    # Optional annotations
    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, format(val, fmt), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return pivot, fig, ax




def extractDataSample(sample: dict):
    videoName = sample["videoName"]
    isVideoViolent = sample["isVideoViolent"]
    conversation = sample["conversation"]
    staySilentProb_list = []

    for conv in conversation[2:]:
        staySilentProb = conv["staySilentProb"]
        staySilentProb_list.append(staySilentProb)
    

    sample_extracted = {
        "videoName": videoName,
        "isVideoViolent": isVideoViolent,
        "staySilentProb_list": staySilentProb_list
    }

    return sample_extracted




def binary_classification_metrics(gt, pred):
    """
    Compute binary classification metrics.

    Args:
        gt (list[int]): Ground truth labels (0 or 1)
        pred (list[int]): Predicted labels (0 or 1)

    Returns:
        dict: accuracy, mean_accuracy (balanced accuracy),
              precision, recall, f1_score
    """
    assert len(gt) == len(pred), "gt and pred must have the same length"

    tp = sum((g == 1 and p == 1) for g, p in zip(gt, pred))
    tn = sum((g == 0 and p == 0) for g, p in zip(gt, pred))
    fp = sum((g == 0 and p == 1) for g, p in zip(gt, pred))
    fn = sum((g == 1 and p == 0) for g, p in zip(gt, pred))

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0

    # Class-wise recall
    recall_pos = tp / (tp + fn) if (tp + fn) else 0  # violence recall
    recall_neg = tn / (tn + fp) if (tn + fp) else 0  # non-violence recall

    mean_accuracy = (recall_pos + recall_neg) / 2

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = recall_pos
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )

    return {
        "accuracy": accuracy,
        "mean_accuracy": mean_accuracy,  # balanced accuracy
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }



# numFrames_decide = if you dont want any wait let it be 1
def getMetrics_numFrames_decide(extractedSamples_list, numFrames_decide, decide_avg_thresh):
    
    groundTruth_list = []
    preditction_list = []

    for extractedSample in extractedSamples_list:

        isVideoViolent = extractedSample["isVideoViolent"]
        gt = 1 if isVideoViolent else 0
        groundTruth_list.append(gt)

        # staySilentprob_window = []  # has len numFrames_decide
        staySilentprob_window = deque(maxlen=numFrames_decide)

        predictionViolent = 0

        for staySilentProb in extractedSample["staySilentProb_list"]:

            if len(staySilentprob_window) < numFrames_decide:
                staySilentprob_window.append(staySilentProb)
            else:
                avg_staySilenProb_window = sum(staySilentprob_window) / len(staySilentprob_window)
                if avg_staySilenProb_window < decide_avg_thresh: # Then Respond, then violent prediction
                    predictionViolent = 1
                    break
                else:
                    staySilentprob_window.append(staySilentProb)


        
        preditction_list.append(predictionViolent)
    

    metrics = binary_classification_metrics(gt = groundTruth_list, pred=preditction_list)
    return metrics

            






# numSeconds_wait = 2
# fps = 4


chkPoint = "train_incNvSegment/chk_132"
# chkPoint = "train_38/chk_84"

setType = "evalSet"
# setType = "test"
# setType = "yt_final"

annotation_file = f"/home/zeidan/Masters/videollm-online_fineTune_mola/demo/eval_mola_stream_results/{chkPoint}/{setType}_results_4fps_thresh_0_analyzeProb.jsonl"


# annotation_file = f"/home/zeidan/Masters/videollm-online_fineTune_mola/demo/eval_mola_stream_results/{chkPoint}/{setType}_results_4fps_thresh_0_analyzeProb.jsonl"

# annotation_file = f"/home/zeidan/Masters/videollm-online_fineTune_mola/demo/eval_mola_stream_results/{chkPoint}/{setType}_results_4fps_thresh_0_analyzeProb.jsonl"



saveHeatmaps = False

with open(annotation_file, "r", encoding="utf-8") as annoFile:
    extractedSamples_list = []
    for line in annoFile:
        sample = json.loads(line)
        sample_extracted = extractDataSample(sample)
        extractedSamples_list.append(sample_extracted)



# numFrames_decide = 4
# decide_avg_thresh = 0.5

bestHyperParams = {
    "f1_score": 0,
    "numFrames_decide": -1,
    "decide_avg_thresh": -1
}
bestMetrics = {}

saved_hyperParam_results = []


for numFrames_decide in range(1,5):
    for decide_avg_thresh in np.arange(0.1,1.0,0.05):
        numFrames_decide = 1
        # decide_avg_thresh = 0.725

        metrics = getMetrics_numFrames_decide(extractedSamples_list=extractedSamples_list, numFrames_decide=numFrames_decide, decide_avg_thresh=decide_avg_thresh)
        hyperparam_res = {
            "numFrames_decide" : numFrames_decide,
            # "decide_avg_thresh" : decide_avg_thresh,
            "decide_avg_thresh" : f"{decide_avg_thresh * 100:.1f}",
            "accuracy": metrics["accuracy"],
            "mean_accuracy": metrics["mean_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        }
        saved_hyperParam_results.append(hyperparam_res)
        # print(metrics)
        if metrics["f1_score"] > bestHyperParams["f1_score"]:
            bestHyperParams["f1_score"] = metrics["f1_score"]
            bestHyperParams["numFrames_decide"] = numFrames_decide
            bestHyperParams["decide_avg_thresh"] = decide_avg_thresh
            bestMetrics = metrics

print("="*5)
print("Best Hyper Params:")
print(bestHyperParams)
print(bestMetrics)
print("="*5)

print(saved_hyperParam_results)


if saveHeatmaps:
    save_hparam_heatmap(results=saved_hyperParam_results, hp1="numFrames_decide", hp2="decide_avg_thresh", metric="mean_accuracy", file_name="heatmap_meanAcc.png", out_root_path=f"/home/zeidan/Masters/videollm-online_fineTune_mola/demo/analyze_analyzeProbResults_saved_graphs/{chkPoint}/{setType}")
    save_hparam_heatmap(results=saved_hyperParam_results, hp1="numFrames_decide", hp2="decide_avg_thresh", metric="precision", file_name="heatmap_precision.png", out_root_path=f"/home/zeidan/Masters/videollm-online_fineTune_mola/demo/analyze_analyzeProbResults_saved_graphs/{chkPoint}/{setType}")
    save_hparam_heatmap(results=saved_hyperParam_results, hp1="numFrames_decide", hp2="decide_avg_thresh", metric="recall", file_name="heatmap_recall.png", out_root_path=f"/home/zeidan/Masters/videollm-online_fineTune_mola/demo/analyze_analyzeProbResults_saved_graphs/{chkPoint}/{setType}")
    save_hparam_heatmap(results=saved_hyperParam_results, hp1="numFrames_decide", hp2="decide_avg_thresh", metric="f1_score", file_name="heatmap_f1_score.png", out_root_path=f"/home/zeidan/Masters/videollm-online_fineTune_mola/demo/analyze_analyzeProbResults_saved_graphs/{chkPoint}/{setType}")
