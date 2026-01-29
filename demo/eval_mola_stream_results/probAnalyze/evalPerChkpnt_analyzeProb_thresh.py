

import json
from pathlib import Path

import numpy as np
from analyze_analyzeProbResults import getMetrics_numFrames_decide, extractDataSample


# metrics = getMetrics_numFrames_decide(extractedSamples_list=extractedSamples_list, numFrames_decide=numFrames_decide, decide_avg_thresh=decide_avg_thresh)

annotation_root_path = Path("/home/zeidan/Masters/videollm-online_fineTune_mola/demo/eval_mola_stream_results/probAnalyze/8fps/train_incNvSegment")
saved_hyperParam_results = []
bestHyperParams = {
    "f1_score": 0,
    "accuracy": 0,
    "numFrames_decide": -1,
    "decide_avg_thresh": -1

}
bestMetrics = {}
annotation_files = sorted(
    annotation_root_path.rglob("*.jsonl"),
    key=lambda p: int(p.parent.name.split("_")[-1])
)
for annotation_file in annotation_files:

    chkPnt = annotation_file.parent.name.split("_")[-1]
    print(annotation_file)

    with open(annotation_file, "r", encoding="utf-8") as annoFile:
            extractedSamples_list = []
            for line in annoFile:
                sample = json.loads(line)
                sample_extracted = extractDataSample(sample)
                extractedSamples_list.append(sample_extracted)



   

    # saved_hyperParam_results = []

    for numFrames_decide in range(1,5):
        for decide_avg_thresh in np.arange(0.1,1.0,0.05):
 
            numFrames_decide = 4
            decide_avg_thresh = 0.725

            metrics = getMetrics_numFrames_decide(extractedSamples_list=extractedSamples_list, numFrames_decide=numFrames_decide, decide_avg_thresh=decide_avg_thresh)
            # print(metrics)
            hyperparam_res = {
                "chkPnt": chkPnt,
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
            print(metrics)
            if metrics["f1_score"] > bestHyperParams["f1_score"]:
                bestHyperParams["f1_score"] = metrics["f1_score"]
                bestHyperParams["accuracy"] = metrics["accuracy"]
                bestHyperParams["numFrames_decide"] = numFrames_decide
                bestHyperParams["decide_avg_thresh"] = decide_avg_thresh
                bestHyperParams["chkPnt"] = chkPnt
                bestMetrics = metrics

print("="*5)
print("Best Hyper Params:")
print(bestHyperParams)
print(bestMetrics)
print("="*5)


# for hyperparam_res in saved_hyperParam_results:
#      print("\n")
#      print(hyperparam_res)