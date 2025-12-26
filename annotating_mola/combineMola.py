
import json
import os
from pathlib import Path
import re
import shutil

from collections import defaultdict



# globalVideoCount = 1

summary = defaultdict(int)

videosToBeRemoved = []


videosStats = defaultdict(list)

annotations = []


def getVideosToBeRemoved():
    global videosToBeRemoved
    for video in videosStats["videosToBeRemoved"]:
        videoName = video["videoName"]
        videosToBeRemoved.append(videoName)


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
        # print(f"Loading {file}")
        filePath = os.path.join(statsDir, file)
        with open(filePath, "r", encoding="utf-8", newline="") as f:
            data = json.load(f)
            videosStats[file.replace(".json","")].extend(data)


# def handleNonViolentVideos(inputDir, outRoot, data): 
#     global globalVideoCount, annotations

#     for i, video in enumerate(data, start=1):
#         videoName = video["videoName"]
#         segment = video["segments"][0]

#         segmentName, numFrames = segment[0], segment[1]


#         newVideoPath =  f"NONVIOLENT/{videoName}_{globalVideoCount}"


#         videoInputDir = os.path.join(inputDir, segmentName)
#         videoOutputDir = os.path.join(outRoot,newVideoPath)

#         annotation = {
#             "videoName": newVideoPath,
#             "numFrames_nonViolent_segment": numFrames,
#             "numFrames_violent_segment": 0,
#             "numFrames_nonViolent_extraSegment":0,
#         }

#         annotations["nonViolent"].append(annotation)

#         src = Path(videoInputDir)
#         dst = Path(videoOutputDir)

#         if not src.is_dir():
#             raise NotADirectoryError(f"Source folder not found: {src}")

#         dst.mkdir(parents=True, exist_ok=False)

#         countFrames = 0
#         for p in src.iterdir():
#             if p.is_file() and p.suffix.lower() == ".jpg":
#                 shutil.copy2(p, dst / p.name)  # copy2 keeps metadata where possible
#                 countFrames += 1

#         if countFrames != numFrames:
#             raise ValueError(f"NumFrames doesnt match the num frames from annotations for segment {segmentName}")
        
#         globalVideoCount += 1

#         print(f"Copied video nonViolent {i} successfully.")

#     print("Finished copying NonViolent Videos")
        

def merge_frames(segment_list, input_dir, out_dir, start=1, digits=5, ext="jpg"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)

    sum_numFrames = 0
    i = start
    for segment in segment_list:  # keeps your list order
        segment_name, segment_numFrames = segment[0], segment[1]
        seg_path = os.path.join(input_dir, segment_name)

        sum_numFrames += segment_numFrames

        frames = sorted(Path(seg_path).glob(f"*.{ext}"), key=lambda p: int(p.stem.split("_")[-1]))
        for f in frames:
            dst = out / f"{i:0{digits}d}.{ext}"
            shutil.copy2(f, dst)  # copy frame
            i += 1

    if i-1 != sum_numFrames:
            print(f"Segments = {segment_list}")
            print(f"numFrames copied = {i-1} & sum_numFrames = {sum_numFrames}")
            raise ValueError(f"NumFrames doesnt match the num frames from annotations for video {out_dir}")
    
    print(f"\ncopied frames of video {out_dir.split("/")[-1]} with {len(segment_list)} segments")
    return sum_numFrames

def sort_segments_by_trailing_number(segments):
    # segments = [[path, count], [path, count], ...]
    def key(item):
        path = item[0]
        m = re.search(r'_(\d+)$', path)   # digits at very end, after last underscore
        if not m:
            print(f"Culdnt extract number from segment {path}")
        return int(m.group(1)) 

    return sorted(segments, key=key)


def handle4Segments(inputDir_root, outDir_root, data): #duplicate videos , different annotations
    splitData = []
    for video in data:
        videoName = video["videoName"]
        print(f"\nHandling 4 segment video {videoName}")
        segment_list = video["segments"]

        sorted_segments = sort_segments_by_trailing_number(segment_list)

        video_1 = {
            "videoName": f"{videoName}_dup_1",
            "segments": sorted_segments[:2]
        }
        video_2 = {
            "videoName": f"{videoName}_dup_2",
            "segments": sorted_segments[2:]
        }
        splitData.append(video_1)
        splitData.append(video_2)

    handleSegments(inputDir_root, outDir_root, splitData)

def handle5Segments(inputDir_root, outDir_root, data): #combine all into 2 segments, assuming atleast 1 violent segment

    for video in data:
        videoName = video["videoName"]
        print(f"\nHandling 5 segment video {videoName}")
        segment_list = video["segments"]

        sorted_segments = sort_segments_by_trailing_number(segment_list)

        currentVideoType = "nonviolent"

        combinedSegments = []

        # combinedSegmentsNumFrames = 0

        for segment in sorted_segments:
            segment_name, segment_numFrames = segment[0], segment[1]
            segment_type = segment_name.split("/")[0].lower()

            if currentVideoType == segment_type:
                combinedSegments.append(segment)
            else:
                tempFolderName_1 = "NONVIOLENT/temp_1"
                out_dir_temp = os.path.join(inputDir_root, tempFolderName_1)
                combinedSegmentsNumFrames = merge_frames(segment_list=combinedSegments, input_dir=inputDir_root, out_dir=out_dir_temp)
                break




        segment_1 = [tempFolderName_1, combinedSegmentsNumFrames]


        secondSegment_list = sorted_segments[len(combinedSegments):]

        tempFolderName_2 = "VIOLENT/temp_1"
        out_dir_temp_2 = os.path.join(inputDir_root, tempFolderName_2)
        combinedSegmentsNumFrames_2 = merge_frames(segment_list=secondSegment_list, input_dir=inputDir_root, out_dir=out_dir_temp_2)

        segment_2 = [tempFolderName_2, combinedSegmentsNumFrames_2]

        videoSegments_list = [segment_1, segment_2]

        print(f"1st segment is comprised from {len(combinedSegments)} and 2nd segment is from {len(secondSegment_list)}")

        videoObj = {
            "videoName": videoName,
            "segments": videoSegments_list
        }

        handleSegments(inputDir_root=inputDir_root, outDir_root=outDir_root, data=[videoObj])

        shutil.rmtree(out_dir_temp)  # deletes folder + everything inside
        shutil.rmtree(out_dir_temp_2)  # deletes folder + everything inside
        print("Deleted temp folders!")



        




def handleSegments(inputDir_root, outDir_root, data):

    global annotations, summary

    for i,video in enumerate(data, start=1):
        videoName = video["videoName"]
        if videoName in videosToBeRemoved:
            print(f"Skipped video {videoName} as it should be removed")
            summary[f"skipped_videos"] += 1
            continue
        segment_list = video["segments"]

        if len(segment_list) == 4:
            handle4Segments(inputDir_root=inputDir_root, outDir_root=outDir_root, data=[video])
            summary[f"Handled_4_segments"] += 1
            continue
        elif len(segment_list) == 5:
            handle5Segments(inputDir_root=inputDir_root, outDir_root=outDir_root, data=[video])
            summary[f"Handled_5_segments"] += 1
            continue

        summary[f"{len(segment_list)}_segments"] += 1

        summary[f"Total_videos"] += 1

        annotation = {
            "videoName": videoName,
            "numFrames_nonviolent_segment": 0,
            "numFrames_violent_segment": 0,
            "numFrames_nonviolent_extraSegment":0,
        }

        segment_list_sorted = sort_segments_by_trailing_number(segment_list)

        currentSegmentType = "nonviolent"

        for segment in segment_list_sorted:

            if currentSegmentType == "nonviolent_extra":
                raise ValueError(f"segment after extra segment in video {videoName}")

            segment_name, segment_numFrames = segment[0], segment[1]
            segment_type = segment_name.split("/")[0].lower()

            if currentSegmentType == segment_type:

                annotation[f"numFrames_{segment_type}_segment"] += segment_numFrames
            else:

                if segment_type == "violent":
                    currentSegmentType = "violent"
                    annotation[f"numFrames_{segment_type}_segment"] += segment_numFrames
                elif segment_type == "nonviolent":
                    currentSegmentType = "nonviolent_extra"
                    annotation[f"numFrames_nonviolent_extraSegment"] += segment_numFrames

        

        video_outDir = os.path.join(outDir_root, videoName)
        merge_frames(segment_list=segment_list_sorted, input_dir=inputDir_root, out_dir=video_outDir)
        annotations.append(annotation)     
        print("\nAnnotation:")
        print(annotation)
        print(f"Saved annotation for video {videoName}")



            

        




# def handleViolent2Segments(inputDir, outRoot, data):
#     global globalVideoCount, annotations



#     for i, video in enumerate(data, start=1):
#         videoName = video["videoName"]
#         segmentList = video["segments"]
#         segment1, segment2 = segmentList[0], segmentList[1]

#         segment1_name, segment1_numFrames = segment1[0], segment1[1]
#         segment2_name, segment2_numFrames = segment2[0], segment2[1]


#         segment1_type = segment1_name.split("/")[0]
#         segment2_type = segment2_name.split("/")[0]


#         if segment1_type == "VIOLENT" or segment2_type == "VIOLENT":


#             if segment1_type == "VIOLENT":
#                 segment_list = [segment2, segment1]
#             elif segment2_type == "VIOLENT":
#                 segment_list = [segment1, segment2]

#             newVideoPath =  f"VIOLENT/{videoName}_{globalVideoCount}"
#             annotation = {
#                 "videoName": newVideoPath,
#                 "numFrames_nonViolent_segment": segment_list[0][1],
#                 "numFrames_violent_segment": segment_list[1][1],
#                 "numFrames_nonViolent_extraSegment":0,
#             }


#         else:
#             segmentNum_segment1 = int(segment1_name.split("_")[-1])
#             segmentNum_segment2 = int(segment2_name.split("_")[-1])

#             if segmentNum_segment1 < segmentNum_segment2:        
#                 segment_list = [segment1, segment2]
#             else:
#                 segment_list = [segment2, segment1]

#             newVideoPath =  f"NONVIOLENT/{videoName}_{globalVideoCount}"

#             annotation = {
#                 "videoName": newVideoPath,
#                 "numFrames_nonViolent_segment": segment_list[0][1] + segment_list[1][1],
#                 "numFrames_violent_segment": 0,
#                 "numFrames_nonViolent_extraSegment":0,
#             }


#         out_dir = os.path.join(outRoot, newVideoPath)
#         merge_frames(segment_list=segment_list, input_dir=inputDir, out_dir=out_dir)
#         annotations["nonViolent"].append(annotation)

#         globalVideoCount += 1

#         print(f"Copied video with 2 segments {i} successfully.")

#     print("Finished copying videos with 2 segments Videos")     



        










statsDir = r"C:\Users\szizo\Desktop\videoLLM-online_finetuning\videollm-online_fineTune_mola\annotating_mola\statsMola"
loadStats(statsDir=statsDir)
print(f"loaded {len(videosStats)} stat files")
for i, (statFile, _) in enumerate(videosStats.items(), start=1):
    print(f"{i}. {statFile}")

getVideosToBeRemoved()
            

inputDir_root = r"C:\Users\szizo\Desktop\testCombine\rawframes"
outDir_root = r"C:\Users\szizo\Desktop\testCombine\merged"

data = videosStats["testCombine"]
print(type(data))
handleSegments(inputDir_root=inputDir_root, outDir_root=outDir_root, data=data)

for annotation in annotations:
    if annotation["numFrames_violent_segment"] > 0 :
        summary[f"Violent_videos"] += 1
    else:
        summary[f"Non-Violent_videos"] += 1

    

print("\n" + "=" * 10)
print("Printing Summary ...")
print("=" * 10, "\n")
for key, value in summary.items():
    print(f"\n{key}: {value}")

print("=" * 10, "\n")