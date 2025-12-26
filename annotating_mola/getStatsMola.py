import json
import os, re, shutil

from collections import defaultdict

from pathlib import Path
import shutil

videosDict = defaultdict(list)

videosToSaveInfo = defaultdict(list)

segmentsSkipped = [] #probably all 1 segment violent videos, no annotation

dataCombined = []

videosDataDistrinutionViolence = defaultdict(int)
videosDataDistrinutionNonViolence = defaultdict(int)

violentVideosWithoutViolentSegment = []
videosDataDistrinution_violentVideosWithoutViolentSegment = defaultdict(int)

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

def combineSegments(annotationPath):
    with open(annotationPath, "r", encoding="utf-8") as annoFile:
        for line in annoFile:
            line = line.strip()
            folderDir, numFrames, category = line.split(" ")
            videoName = extractVideoName(segmentName=folderDir)
            videosDict[videoName].append((folderDir, int(numFrames)))





def saveVideosInfo(outDir):

    for info, data in videosToSaveInfo.items():

        print(f"info = {info}...")

        outFile = os.path.join(outDir, f"{info}.json")

        with open(outFile, "w", encoding="utf-8", newline="") as outF:
            allObjs = []
            for video, segmentList in data:
        
                jsonObj = {
                    "videoName": video,
                    "segments": [segment for segment in segmentList]
                }
                allObjs.append(jsonObj)

            json.dump(allObjs, outF, ensure_ascii=False, indent=4)


        


def getVideosData():

    for video, segmentList in videosDict.items():
        isViolent = isVideoViolent(video)
        if isViolent:
            videosDataDistrinutionViolence[len(segmentList)] += 1

            videosToSaveInfo[f"violent_{len(segmentList)}"].append((video, segmentList))


            foundViolentSegment = False
            for segment, _ in segmentList:
                if segment.split("/")[0] == "VIOLENT":
                    foundViolentSegment = True
                    break

            if not foundViolentSegment:
                violentVideosWithoutViolentSegment.append((video, segmentList))
                videosToSaveInfo["violent_withNoViolentSegment"].append((video, segmentList))
                videosDataDistrinution_violentVideosWithoutViolentSegment[len(segmentList)] += 1



        else:
            videosDataDistrinutionNonViolence[len(segmentList)] += 1
            videosToSaveInfo[f"non-violent_vids"].append((video, segmentList))
    
    print("\n"+ f"=" *10)
    print(f"Printing distribution for violent videos")
    print(f"=" *10)


    for numSegments, count in videosDataDistrinutionViolence.items():
        print(f"Violent Videos with {numSegments} segments = {count}")


    print("\n"+ f"=" *10)
    print(f"Printing distribution for non-violent videos")
    print(f"=" *10)


    for numSegments, count in videosDataDistrinutionNonViolence.items():
        print(f"Non-Violent Videos with {numSegments} segments = {count}")

    
    

    print("\n"+ f"=" *10)
    print(f"Found - {len(violentVideosWithoutViolentSegment)} - videos without violent segment")
    for numSegments, count in videosDataDistrinution_violentVideosWithoutViolentSegment.items():
        print(f"Violent Videos with {numSegments} segments = {count} has no violent segment")
    print(f"=" *10)



videosToBeRemovedPath = Path(r"C:\Users\szizo\Desktop\videoLLM-online_finetuning\videollm-online_fineTune_mola\annotating_mola\statsMola\videosToBeRemoved.json")
            
for annoType in ["total", "train", "val", "test"]:
    annotationFilePath = rf"C:\Users\szizo\Desktop\videoLLM-online_finetuning\videollm-online_fineTune_mola\annotating_mola\InCar_GT_annotations\recheck_INCAR2c_{annoType}_rawframes.txt"
    combineSegments(annotationPath=annotationFilePath)
    print(f"Length of dict = {len(videosDict)}")
    getVideosData()
    outDirSaveVideosInfo = rf"C:\Users\szizo\Desktop\videoLLM-online_finetuning\videollm-online_fineTune_mola\annotating_mola\statsMola\{annoType}"
    if not os.path.exists(outDirSaveVideosInfo):
        os.makedirs(outDirSaveVideosInfo)
    saveVideosInfo(outDir=outDirSaveVideosInfo)
    shutil.copy2(videosToBeRemovedPath, Path(outDirSaveVideosInfo) / videosToBeRemovedPath.name) 








