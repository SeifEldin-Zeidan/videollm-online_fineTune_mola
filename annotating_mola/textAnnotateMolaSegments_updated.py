import json
import re


scenariosLabels = {
    "C1": {
        "non_violent": "Discuss (P1,P2)",
        "violent": "Pushing (P1) - Punching (P1)",
    },
    "C2": {
        "non_violent": "Singing/dancing (P1) - Asking for a kiss (P1) - Refusing kiss (P2)",
        "violent": "Slapping (P1) - Pulling/Pushing (P1)",
    },
    "C3": {
        "non_violent": "Arguing (P1,P2) - Showing middle finger (P2)",
        "violent": "Kicking (P1) - Strangling (P1)",
    },
    "C4": {
        "non_violent": "Arguing (P1,P2) - Threatening to hit (P2) - Provoking (P1)",
        "violent": "Slapping (P1)",
    },
    "C5": {
        "non_violent": "Approaching/Coming closer to the other person (P1) - Stroking the hair (P1) - Touching on the body (P1)",
        "violent": "Beating with backpack/purse (P2)",
    },
    "C6": {
        "non_violent": "Greeting/complimenting (P1,P2) - Showing something on phone (P1)",
        "violent": "Threatening with scissors (P1) - Stealing/asking for the wallet (P1)",
    },
    "C7": {
        "non_violent": "Arguing (P1,P2)",
        "violent": "Picking a gun from backpack/purse/clothes (P1) - Pointing a gun (P1) - Punching with the gun (P1)",
    },
    "C8": {
        "non_violent": "Arguing (P1,P2)",
        "violent": "Picking knife from backpack/purse/clothes (P1) - Pointing knife (P1) - Stabbing (P1)",
    },
    "C9": {
        "non_violent": "Coming closer to the other person (P1)",
        "violent": "Threatening with knife (P1) - Touching on the body (P1)",
    },
    "C10": {
        "non_violent": "Playing on the phone (P1) - Playing on the phone (P2)",
        "violent": "Slap (s) (P1) - Holding and punching (P2)",
    },
    "C11": {
        "non_violent": "Sleeping (P2) - Drinking (P1)",
        "violent": "Throwing bottle (P1) - Pushing (P2)",
    },
    "C12": {
        "non_violent": "Playing on the phone (P2) - Moving closer/ looking at the cell phone (P1) - Pulling away (P2)",
        "violent": "Pulling/Pushing (P1)",
    },
}


scenarios_actionlabels = {    #Categories
    1: {
        "non_violent": "Verbal Interaction",
        "violent": "Physical Assault",
    },
    2: {
        "non_violent": "Individual Activity",
        "violent": "Physical Assault",  ####FIXED SPELLING, REGENERATE
    },
    3: {
        "non_violent": "Verbal Interaction",
        "violent": "Physical Assault",
    },
    4: {
        "non_violent": "Verbal Interaction",
        "violent": "Physical Assault",
    },
    5: {
        "non_violent": "Approaching",
        "violent": "Physical Assault",
    },
    6: {
        "non_violent": "Social Interaction",   
        "violent": "Sharp Object",
    },
    7: {
        "non_violent": "Verbal Interaction",
        "violent": "Gun",
    },
    8: {
        "non_violent": "Verbal Interaction",
        "violent": "Sharp Object",
    },
    9: {
        "non_violent": "Approaching",
        "violent": "Sharp Object",
    },
    10: {
        "non_violent": "Individual Activity",  
        "violent": "Physical Assault",
    },
    11: {
        "non_violent": "Individial Activity",   
        "violent": "Physical Assault",
    },
    12: {
        "non_violent": "Approaching",   #NOTE check confusion matrix #################
        "violent": "Physical Assault",
    },
    13: {
        "non_violent": "Social Interaction", 
    },
    14: {
        "non_violent": "Social Interaction", 
    },
    15: {
        "non_violent": "Individual Activity",
    },
    16: {
        "non_violent": "Individual Activity",
    },
    17: {
        "non_violent": "Individual Activity",
    },
    18: {
        "non_violent": "Individual Activity",
    },
    19: {
        "non_violent": "Individual Activity",
    },
    20: {
        "non_violent": "Individual Activity",
    },
}



scenarios_nl = {
    1: {
        "non_violent": "The left passenger and the right passenger are talking.",
        "violent": "The left passenger pushes and then punches the right passenger.",
    },
    2: {
        "non_violent": "The left passenger is singing and dancing, then asks the right passenger for a kiss, but it's refused.",
        "violent": "The left passenger slaps and then pulls and pushes the right passenger.",
    },
    3: {
        "non_violent": "The left passenger and the right passenger are arguing, and the right passenger shows the middle finger.",
        "violent": "The left passenger kicks and then strangles the right passenger.",
    },
    4: {
        "non_violent": "The left passenger and the right passenger are arguing; the right passenger threatens to hit, and The left passenger provokes.",
        "violent": "The left passenger slaps the right passenger.",
    },
    5: {
        "non_violent": "The left passenger approaches the right passenger, strokes hair, and touches body.",
        "violent": "the right passenger beats The left passenger with a bag.",
    },
    6: {
        "non_violent": "The left passenger and the right passenger greet each other, and then The left passenger shows something to on the phone.",
        "violent": "The left passenger threatens the right passenger with scissors and then demands wallet.",
    },
    7: {
        "non_violent": "The left passenger and the right passenger are arguing.",
        "violent": "The left passenger takes out a gun, points it at the right passenger, and then hits with the gun.",
    },
    8: {
        "non_violent": "The left passenger and the right passenger are arguing.",
        "violent": "The left passenger takes out a knife, points it at the right passenger, and then stabs.",
    },
    9: {
        "non_violent": "The left passenger comes closer to the right passenger.",
        "violent": "The left passenger threatens the right passenger with a knife and then touches body.",
    },
    10: {
        "non_violent": "The left passenger and the right passenger are playing on the phone.",
        "violent": "The left passenger slaps the right passenger, and then the right passenger holds and punches The left passenger.",
    },
    11: {
        "non_violent": "the right passenger is sleeping while The left passenger is drinking.",
        "violent": "The left passenger throws a bottle at the right passenger, and then the right passenger pushes.",
    },
    12: {
        "non_violent": "the right passenger is playing on the phone; The left passenger moves closer to look at the phone, and then the right passenger pulls away.",
        "violent": "The left passenger pulls and pushes the right passenger.",
    },
    13: {
        "non_violent": "The left passenger and the right passenger are talking; the right passenger begins to cry, and then they hug.",
    },
    14: {
        "non_violent": "The left passenger asks the right passenger to take pictures; the right passenger takes pictures, and then shows the pictures.",
    },
    15: {
        "non_violent": "The left passenger applies lipstick and fixes hair, while the right passenger is sleeping.",
    },
    16: {
        "non_violent": "The left passenger sneezes, takes out a tissue, and wipes nose, while the right passenger is reading a book.",
    },
    17: {
        "non_violent": "The left passenger yawns and stretches, while the right passenger puts on headphones, listens to music.",
    },
    18: {
        "non_violent": "The left passenger is eating and drinking, while the right passenger is taking pictures on a phone.",
    },
    19: {
        "non_violent": "The left passenger talks on the phone, while the right passenger is typing on laptop and coughs.",
    },
    20: {
        "non_violent": "The left passenger is writing on a notepad, while the right passenger applies alcohol gel.",
    },

}


scenarios_nl_updated = {
    1: {
        "non_violent": "The left passenger and the right passenger are talking.",
        "violent": "The left passenger pushes and then punches the right passenger.",
    },
    2: {
        "non_violent": "The left passenger is singing and dancing while the right passenger refuses a kiss.",
        "violent": "The left passenger slaps, pulls, and pushes the right passenger.",
    },
    3: {
        "non_violent": "The left passenger and the right passenger are arguing and the right passenger shows the middle finger.",
        "violent": "The left passenger kicks and then strangles the right passenger.",
    },
    4: {
        "non_violent": "The left passenger and the right passenger are arguing while the right passenger threatens the left passenger.",
        "violent": "The left passenger slaps the right passenger.",
    },
    5: {
        "non_violent": "The left passenger strokes the hair and touches the body of the right passenger.",
        "violent": "The right passenger beats the left passenger with a bag.",
    },
    6: {
        "non_violent": "The left passenger and the right passenger greet each other and look at a phone.",
        "violent": "The left passenger threatens the right passenger with scissors and demands their wallet.",
    },
    7: {
        "non_violent": "The left passenger and the right passenger are engaged in a verbal argument.",
        "violent": "The left passenger points a gun at the right passenger and then hits them with the gun.",
    },
    8: {
        "non_violent": "The left passenger and the right passenger are having a heated argument.",
        "violent": "The left passenger points a knife at the right passenger and then stabs them.",
    },
    9: {
        "non_violent": "The left passenger moves closer to the right passenger.",
        "violent": "The left passenger threatens the right passenger with a knife and touches their body.",
    },
    10: {
        "non_violent": "The left passenger and the right passenger are both playing on their phones.",
        "violent": "The left passenger slaps the right passenger, and then the right passenger holds and punches the left passenger.",
    },
    11: {
        "non_violent": "The right passenger is sleeping while the left passenger is drinking from a bottle.",
        "violent": "The left passenger throws a bottle at the right passenger, and the right passenger pushes them back.",
    },
    12: {
        "non_violent": "The right passenger is playing on a phone while the left passenger moves closer to look.",
        "violent": "The left passenger pulls and pushes the right passenger.",
    },
    13: {
        "non_violent": "The left passenger and the right passenger are talking until the right passenger cries and they hug.",
    },
    14: {
        "non_violent": "The left passenger asks the right passenger to take pictures with a phone and then they look at the photos.",
    },
    15: {
        "non_violent": "The left passenger applies lipstick and fixes their hair while the right passenger sleeps.",
    },
    16: {
        "non_violent": "The left passenger sneezes and wipes their nose while the right passenger reads a book.",
    },
    17: {
        "non_violent": "The left passenger yawns and stretches while the right passenger listens to music with headphones.",
    },
    18: {
        "non_violent": "The left passenger is eating and drinking while the right passenger takes pictures with a phone.",
    },
    19: {
        "non_violent": "The left passenger talks on the phone while the right passenger types on a laptop and coughs.",
    },
    20: {
        "non_violent": "The left passenger writes on a notepad while the right passenger applies alcohol gel to their hands.",
    },
}

subject_gender = {
    "P1": ["The Man", "he", "him", "his"],
    "P2": ["The Woman", "she", "her", "her"],
    "P3": ["The Man", "he", "him", "his"],
    "P4": ["The Woman", "she", "her", "her"],
    "P5": ["The Man", "he", "him", "his"],
    "P6": ["The Man", "he", "him", "his"],
    "P7": ["The Man", "he", "him", "his"],
    "P8": ["The Woman", "she", "her", "her"],
    "P9": ["The Woman", "she", "her", "her"],
    "P10": ["The Man", "he", "him", "his"],
    "P11": ["The Woman", "she", "her", "her"],
    "P12": ["The Man", "he", "him", "his"],
    "P13": ["The Woman", "she", "her", "her"],
    "P14": ["The Woman", "she", "her", "her"],
    "P15": ["The Man", "he", "him", "his"],
    "P16": ["The Man", "he", "him", "his"],
}

# 1) Natural-language sentences (non-violent only)
scenarios_nl_nv_only = {
    13: {
        "non_violent": "P1 and P2 are talking; P2 begins to cry, and then they hug.",
    },
    14: {
        "non_violent": "P1 asks P2 to take pictures; P2 takes pictures, and then <P2sp> shows <P1op> the pictures.",
    },
    15: {
        "non_violent": "P1 applies lipstick and fixes <p1op> hair, while P2 is sleeping.",
    },
    16: {
        "non_violent": "P1 sneezes, takes out a tissue, and wipes <p1op> nose, while P2 is reading a book.",
    },
    17: {
        "non_violent": "P1 yawns and stretches, while P2 puts on headphones, listens to music.",
    },
    18: {
        "non_violent": "P1 is eating and drinking, while P2 is taking pictures on a phone.",
    },
    19: {
        "non_violent": "P1 talks on the phone, while P2 is typing on <P2op> laptop and coughs.",
    },
    20: {
        "non_violent": "P1 is writing on a notepad, while P2 applies alcohol gel.",
    },
}

# 2) Action labels (non-violent only)
scenarios_actionlabels_nv_only = {
    "C13": {
        "non_violent": "talking, crying, hugging",
    },
    "C14": {
        "non_violent": "photographing",
    },
    "C15": {
        "non_violent": "lipstick, hairfixing, sleeping",
    },
    "C16": {
        "non_violent": "sneezing, wiping, reading",
    },
    "C17": {
        "non_violent": "yawning, listening to music",
    },
    "C18": {
        "non_violent": "eating, drinking, photographing",
    },
    "C19": {
        "non_violent": "phoneuse, typing, coughing",
    },
    "C20": {
        "non_violent": "writing, sanitizing",
    },
}


def getScenarioFromVideo(videoPath):
    return re.findall(r"C\d+", videoPath)[0]


def getSegmentDetails(videoPath):
    scenario = getScenarioFromVideo(videoPath)
    # _PATTERN = re.compile(r'\b(P\d+_P\d+)\b')
    # _PATTERN = re.compile(r'(?:^|_)(P\d+)_(P\d+)(?:_|$)')
    _PATTERN = re.compile(r'(?<=_)(P\d+)_(P\d+)(?=_)')
    persons = _PATTERN.search(videoPath)
    # print(videoPath)
    # print(persons)
    person1 = persons.group(0).split('_')[0]
    person2 = persons.group(0).split('_')[1]
    person1List = subject_gender[person1]
    person2List = subject_gender[person2]
    return scenario, person1List, person2List



def formulateNLAnnotation(videoPath, category, flip):
    scenario, person1List, person2List = getSegmentDetails(videoPath)
 
    scenarionInt = int(scenario[1:])
    scenarioObj = scenarios_nl[scenarionInt]
    if category == "0":
        sentence = scenarioObj["violent"]
        
    elif category == "1":
        sentence = scenarioObj["non_violent"]
        

    if person1List[0] == person2List[0]:
        if not flip:
            person1Name = person1List[0] + " on the left"
            person2Name = person2List[0] + " on the right"
        else:
            person1Name = person1List[0] + " on the right"
            person2Name = person2List[0] + " on the left"

        
    else:
        person1Name = person1List[0]
        person2Name = person2List[0]

    sentenceUpdated = sentence
    sentenceUpdated = re.sub(r'<P1sp>', person1List[1], sentenceUpdated,  flags=re.IGNORECASE)
    sentenceUpdated = re.sub(r'<P1op>', person1List[2], sentenceUpdated,  flags=re.IGNORECASE)
    sentenceUpdated = re.sub(r'<P1pp>', person1List[3], sentenceUpdated,  flags=re.IGNORECASE)
    sentenceUpdated = re.sub(r'P1', person1Name, sentenceUpdated,  flags=re.IGNORECASE)

    sentenceUpdated = re.sub(r'<P2sp>', person2List[1], sentenceUpdated,  flags=re.IGNORECASE)
    sentenceUpdated = re.sub(r'<P2op>', person2List[2], sentenceUpdated,  flags=re.IGNORECASE)
    sentenceUpdated = re.sub(r'<P2pp>', person2List[3], sentenceUpdated,  flags=re.IGNORECASE)
    sentenceUpdated = re.sub(r'P2', person2Name, sentenceUpdated,  flags=re.IGNORECASE)

    if category == "0":
        actionLabel = scenarios_actionlabels[scenarionInt]["violent"]
        vDet = "Yes"
       
        attacker = person1Name

        if scenario == "C5":
            attacker = person2Name
        
        if scenario in ["C10", "C11"]:
            attacker = person1Name + " and " + person2Name


    elif category == "1":
        actionLabel = scenarios_actionlabels[scenarionInt]["non_violent"]
        vDet = "No"
        attacker = "None"

   

    attacker = attacker.capitalize()
    actionLabel = actionLabel.capitalize()
    sentenceUpdated = sentenceUpdated.capitalize()




    annotation = f"Violence Detected: {vDet}\nDescription: {sentenceUpdated}\nAttacker: {attacker}\nAction: {actionLabel}"
    return annotation





def formulateNLAnnotationUpdated(videoPath, category, flip, sentenceObj=None):

    if sentenceObj == None:
        scenario, person1List, person2List = getSegmentDetails(videoPath)
        scenarioInt = int(scenario[1:])
        scenarioAnnotaion = scenarios_nl_updated[scenarioInt]
        if category == "0":
            sentence = scenarioAnnotaion["violent"]
            action_label = scenarios_actionlabels[scenarioInt]["violent"]
            vDet = "Yes"
        elif category == "1":
            sentence = scenarioAnnotaion["non_violent"]
            action_label = scenarios_actionlabels[scenarioInt]["non_violent"]
            vDet = "No"
    else:
        scenario = sentenceObj["scenario"]
        sentence = sentenceObj["sentence"]
        action_label = sentenceObj["action_label"]
        vDet = sentenceObj["vDet"]

    


    if flip:
        temp = re.sub(r"left passenger", "<temp>", sentence, flags=re.IGNORECASE)
        temp = re.sub(r"right passenger", "left passenger", temp, flags=re.IGNORECASE)
        sentence = re.sub(r"<temp>", "right passenger", temp, flags=re.IGNORECASE)

    

    if category == "0":
        attacker = "The left passenger"
        if flip:
            attacker = "The right passenger"

        if scenario in ["C10", "C11"]:
            attacker = "Uncertain"
    elif category == "1":
        attacker = "None"

    
    attacker = attacker.capitalize()
    action_label = action_label.capitalize()
    sentence = sentence.capitalize()


    annotation = f"Violence Detected: {vDet}\nDescription: {sentence}\nAttacker: {attacker}\nCategory: {action_label}"
    return annotation





def annotateSegmentsWithNL(annotationFilePath, outputFilePath):
    countUnknown3rd = 0
    countRemovedNoAnnotation = 0
    data = []
    with open(annotationFilePath, 'r') as infile:
        it = iter(infile)
        for line in it:
            videoPath, numFrames, category = line.strip().split(' ')
            videoPathKey = "_".join(videoPath.split("/")[-1].split("_")[:-1])
            scenario = getScenarioFromVideo(videoPathKey)

            annotation = formulateNLAnnotationUpdated(videoPath, category, False)
            annotationFlipped = formulateNLAnnotationUpdated(videoPath, category, True)

            # print(f"annotation:\n {annotation}")
            # print(f"\nannotation flipped:\n {annotationFlipped}\n")
            # input()

            sample = {
                "videoPath": videoPath,
                "numFrames": numFrames,
                "category": category,
                "annotation": annotation,
                "annotation_flipped": annotationFlipped,
                "scenario": scenario
            }
            # print(sample)
            # input()
            
            if sample["videoPath"] in segments3rd_toRemove:
                sample["annotation"] = sample["annotation_flipped"] = "unknown"
                countUnknown3rd += 1
                if fileType == "train":   #####remove unknown annotation from trainning, the 39 3rd segments
                    continue
            
            if sample["videoPath"] in videosWithoutAnnotation:
                countRemovedNoAnnotation += 1
                continue

            data.append(sample)

        print(f"Unkown: {countUnknown3rd}")
        print(f"Removed No Annotation: {countRemovedNoAnnotation}")

    with open(outputFilePath, "w", encoding="utf-8") as outfile:
        for s in data:
            outfile.write(json.dumps(s, ensure_ascii=False) + "\n")





if __name__ == "__main__":

    segments3rd_toRemove = set()
    with open("/home/zeidan/Masters/mola_split_mmaction/segmentsWithVideosWith3Segments_total.txt", "r") as segmentsOfVideosWith3SegmentsFile:
        lines3rdSegments = segmentsOfVideosWith3SegmentsFile.readlines()


    for i, line in enumerate(lines3rdSegments, start=1):
        if i % 3 == 0:
            videoPath , _, _ = line.split(" ")
            segments3rd_toRemove.add(videoPath)
            # print(f"removing line {line}")
        
    videosWithoutAnnotation = set()

    with open("/home/zeidan/Masters/mola_split_mmaction/videosWithoutAnnotaions.txt", "r") as videosWithoutAnnotationFile:
        videoPathWithoutAnnotationList = videosWithoutAnnotationFile.readlines()

    for videoPath in videoPathWithoutAnnotationList:
        videosWithoutAnnotation.add(videoPath.strip())




    fileType = "test"
    annotationFilePath = f"/home/zeidan/Masters/mola_split_mmaction/InCar_GT_annotations/recheck_INCAR2c_{fileType}_rawframes.txt"
    outputFilePath = f"/home/zeidan/Masters/mola_split_mmaction/InCar_GT_annotations/updated1_naturalLanguageAnnotationSegmentsAll/nlAnnotations_{fileType}.jsonl"
    annotateSegmentsWithNL(annotationFilePath, outputFilePath)