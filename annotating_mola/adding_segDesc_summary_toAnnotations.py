import json, os, re
from pathlib import Path

from textAnnotateMolaSegments_updated import scenariosLabels, scenarios_actionlabels, scenarios_nl, scenarios_nl_updated, subject_gender, getScenarioFromVideo, getSegmentDetails, formulateNLAnnotationUpdated

scenarios_nl_updated_summary = {
    1: "The left passenger and the right passenger are talking when the left passenger pushes and then punches the right passenger.",
    2: "The left passenger is singing and dancing while the right passenger refuses a kiss, after which the left passenger slaps, pulls, and pushes the right passenger.",
    3: "The left passenger and the right passenger are arguing and the right passenger shows the middle finger before the left passenger kicks and then strangles the right passenger.",
    4: "The left passenger and the right passenger are arguing while the right passenger threatens the left passenger, leading the left passenger to slap the right passenger.",
    5: "The left passenger strokes the hair and touches the body of the right passenger until the right passenger beats the left passenger with a bag.",
    6: "The left passenger and the right passenger greet each other and look at a phone before the left passenger threatens the right passenger with scissors and demands their wallet.",
    7: "The left passenger and the right passenger are engaged in a verbal argument when the left passenger points a gun at the right passenger and then hits them with the gun.",
    8: "The left passenger and the right passenger are having a heated argument before the left passenger points a knife at the right passenger and then stabs them.",
    9: "The left passenger moves closer to the right passenger while threatening the right passenger with a knife and touching their body.",
    10: "The left passenger and the right passenger are both playing on their phones when the left passenger slaps the right passenger, and then the right passenger holds and punches the left passenger.",
    11: "The right passenger is sleeping while the left passenger is drinking from a bottle, after which the left passenger throws a bottle at the right passenger and the right passenger pushes them back.",
    12: "The right passenger is playing on a phone while the left passenger moves closer to look and then pulls and pushes the right passenger.",
    13: "The left passenger and the right passenger are talking until the right passenger cries and they hug.",
    14: "The left passenger asks the right passenger to take pictures with a phone and then they look at the photos.",
    15: "The left passenger applies lipstick and fixes their hair while the right passenger sleeps.",
    16: "The left passenger sneezes and wipes their nose while the right passenger reads a book.",
    17: "The left passenger yawns and stretches while the right passenger listens to music with headphones.",
    18: "The left passenger is eating and drinking while the right passenger takes pictures with a phone.",
    19: "The left passenger talks on the phone while the right passenger types on a laptop and coughs.",
    20: "The left passenger writes on a notepad while the right passenger applies alcohol gel to their hands.",
}



def getScenarioFromVideo_num(videoPath):
    return int(re.findall(r"C\d+", videoPath)[0][1:])


def format_summary(sample):
    category = "0" if sample["numFrames_violent_segment"] > 0 else "1"

    action_label = sample["violent_seg_label"] if sample["violent_seg_label"] else sample["non_violent_seg_label"]
    vDet = "Yes" if category == "0" else "No"

    #    scenario = sentenceObj["scenario"]
    #     sentence = sentenceObj["sentence"]
    #     action_label = sentenceObj["action_label"]
    #     vDet = sentenceObj["vDet"]

    sentenceObj = {
        "scenario" : getScenarioFromVideo(sample["videoName"]),
        "sentence" : sample["summary"],
        "action_label" : action_label,
        "vDet" : vDet
    }
    summary_formatted = formulateNLAnnotationUpdated(videoPath=None, category=category, flip = False, sentenceObj=sentenceObj)
    summary_formatted_flipped = formulateNLAnnotationUpdated(videoPath=None, category=category, flip = True, sentenceObj=sentenceObj)


    temp = re.sub(r"left passenger", "<temp>", sample["summary"], flags=re.IGNORECASE)
    temp = re.sub(r"right passenger", "left passenger", temp, flags=re.IGNORECASE)
    summary_flipped = re.sub(r"<temp>", "right passenger", temp, flags=re.IGNORECASE)

    return summary_flipped, summary_formatted, summary_formatted_flipped


annotations_dir_path = Path("/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withSampledNumFrames")

out_annotations_dir_path = Path("/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withSampledNumFrames_segDescSummary")
out_annotations_dir_path.mkdir(parents=True, exist_ok=True)

for anno_file in annotations_dir_path.glob("*.json"):
    updated_samples = []
    with open(anno_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
        print(f"Loaded annoFile: {anno_file.name}")
    for sample in samples:
        sample_name = sample["videoName"]
        scenario = getScenarioFromVideo_num(sample_name)

        scenario_desc = scenarios_nl_updated[scenario]
        scenario_label = scenarios_actionlabels[scenario]
        

        sample["non_violent_seg_desc"] = scenario_desc["non_violent"]
        sample["non_violent_seg_label"] = scenario_label["non_violent"]

        if "violent" in scenario_desc:
            sample["violent_seg_desc"] = scenario_desc["violent"]
            sample["violent_seg_label"] = scenario_label["violent"]
        else:
            sample["violent_seg_desc"] = ""
            sample["violent_seg_label"] = ""

        sample["summary"] = scenarios_nl_updated_summary[scenario]
        updated_samples.append(sample)

        summary_flipped, summary_formatted, summary_formatted_flipped = format_summary(sample)

        sample["summary_flipped"] = summary_flipped
        sample["summary_formatted"] = summary_formatted
        sample["summary_formatted_flipped"] = summary_formatted_flipped

        

    out_anno_file_path = out_annotations_dir_path / anno_file.name
    with open(out_anno_file_path, "w", encoding="utf-8") as out_f:
        json.dump(updated_samples, out_f, ensure_ascii=False, indent=4)
        print(f"Saved updated annoFile to: {out_anno_file_path}")