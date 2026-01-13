import os, json

import matplotlib.pyplot as plt
import statistics

def plot_time_distribution(times, bins=20):
    """
    Plots a histogram showing the distribution of times in seconds.

    Parameters:
    - times (list of float): List of time values in seconds
    - bins (int): Number of histogram bins (default: 20)
    """
    if not times:
        raise ValueError("The list of times is empty.")

    plt.figure()
    plt.hist(times, bins=bins)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Times, Total Samples: {len(times)}")
    plt.show()


annotationsTotal_path = "/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withSampledNumFrames_segDescSummary/vLLMonline_mola_total.json"

with open(annotationsTotal_path, "r", encoding="utf-8") as f:
    samples = json.load(f)

violent_triggerTimes = []

for sample in samples:
    if sample["numFrames_violent_segment"] > 0:
        violent_triggerTimes.append(sample["numFrames_nonviolent_segment"]/30)


plot_time_distribution(violent_triggerTimes)


average = statistics.mean(violent_triggerTimes)
mode = statistics.mode(violent_triggerTimes)

print(f"Mean Trigger Time: {average}")
print(f"Mean Percentage of video 20 Secs: {average/20}")
print(f"Mode Trigger Time: {mode}")