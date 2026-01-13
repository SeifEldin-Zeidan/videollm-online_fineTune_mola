import os, json, re

def get_isVideoViolent(videoPath):
    scenario =  int(re.findall(r"C\d+", videoPath)[0][1:])
    if scenario > 12:
        return False
    return True


annoPath = "/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withRandomFps_withSampledNumFrames/vLLMonline_mola_test.json"

with open(annoPath, "r", encoding="utf-8") as f:
    data = json.load(f)


for sample in data:
    isViolent = get_isVideoViolent(sample["videoName"])
    numFrames_violent_segment = sample["numFrames_violent_segment"]

    isViolent_numFrames = numFrames_violent_segment > 0

    if isViolent != isViolent_numFrames:
        print(sample["videoName"])