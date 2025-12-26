
import json
import os
from pathlib import Path
import re
import shutil

from collections import defaultdict



globalVideoCount = 1


videosStats = defaultdict(list)

annotations = defaultdict(list)


def getScenario(video):
    m = re.search(r"_C(\d+)_", video)
    scenario = m.group(1) if m else None
    if scenario == None:
        raise ValueError(f"Couldnt extract scenario for video: {video}")
    return scenario

def isVideoViolent(video):
    scenarioNum =int( getScenario(video=video))
    if scenarioNum <= 12:
        return True
    return False

def extractVideoName(segmentName):
    # VIOLENT/INCAR_20210430_Session_1_C1_P10_P11_2_rgb_625
    videoName = "_".join(segmentName.split("/")[-1].split("_")[:-1])
    return videoName


def loadStats(statsDir):
    global videosStats
    for file in os.listdir(statsDir):
        if not file.lower().endswith(".json"):
            continue
        filePath = os.path.join(statsDir, file)
        with open(filePath, "r", encoding="utf-8", newline="") as f:
            data = json.load(f)
            videosStats[file.replace(".json","")].append(data)


def handleNonViolentVideos(inputDir, outputDir, data): 
    global globalVideoCount, annotations

    for i, video in enumerate(data, start=1):
        videoName = video["videoName"]
        segment = video["segments"][0]

        segmentName, numFrames = segment[0], segment[1]


        newVideoPath =  f"NONVIOLENT/{video}_{globalVideoCount}"


        videoInputDir = os.path.join(inputDir, segmentName)
        videoOutputDir = os.path.join(outputDir,newVideoPath)

        annotation = {
            "videoName": newVideoPath,
            "numFrames_nonViolent_segment": numFrames,
            "numFrames_violent_segment": 0,
            "numFrames_nonViolent_extraSegment":0,
        }

        annotations["nonViolent"].append(annotation)

        src = Path(videoInputDir)
        dst = Path(videoOutputDir)

        if not src.is_dir():
            raise NotADirectoryError(f"Source folder not found: {src}")

        dst.mkdir(parents=True, exist_ok=False)

        countFrames = 0
        for p in src.iterdir():
            if p.is_file() and p.suffix.lower() == ".jpg":
                shutil.copy2(p, dst / p.name)  # copy2 keeps metadata where possible
                countFrames += 1

        if countFrames != numFrames:
            raise ValueError(f"NumFrames doesnt match the num frames from annotations for segment {segmentName}")
        
        globalVideoCount += 1

        print(f"Copied video nonViolent {i} successfully.")

    print("Finished copying NonViolent Videos")
        




def handleViolent2Segments(inputDir, outputDir, data):
    global globalVideoCount, annotations



    for i, video in enumerate(data, start=1):
        videoName = video["videoName"]
        segmentList = video["segments"]
        segment1, segment2 = segmentList[0], segmentList[1]

        segment1_name, segment1_numFrames = segment1[0], segment1[1]
        segment2_name, segment2_numFrames = segment2[0], segment2[1]


        segment1_type = segment1_name.split("/")[0]
        segment2_type = segment2_name.split("/")[0]

        if segment1_type == "VIOLENT" or segment2_type == "VIOLENT":
            



        










statsDir = r"C:\Users\szizo\Desktop\videoLLM-online_finetuning\videollm-online_fineTune_mola\annotating_mola\statsMola"
loadStats(statsDir=statsDir)
print(f"loaded {len(videosStats)} stat files")
for i, (statFile, _) in enumerate(videosStats.items(), start=1):
    print(f"{i}. {statFile}")
            

    


