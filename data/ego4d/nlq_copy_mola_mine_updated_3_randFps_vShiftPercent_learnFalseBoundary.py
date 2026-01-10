
import math
import os, json, collections, torch, tqdm, random
from transformers import EvalPrediction

from ..ego4d import Ego4D
from ..stream import StreamMixIn
from ..utils import temporal_iou, DictWithTo, ceil_time_by_fps



# class StreamMixIn(torch.utils.data.Dataset):
#     def __init__(self, is_training: bool, system_prompt: str, augmentation: bool, max_num_frames: int, tokenizer: PreTrainedTokenizer, **kwargs):
#         super().__init__()
#         self.is_training = is_training
#         self.system_prompt = system_prompt
#         self.augmentation = augmentation
#         self.tokenizer = tokenizer
#         self.max_num_frames = max_num_frames

class NLQ_MOLA(StreamMixIn):

    evaluation_kwargs = DictWithTo(evaluator='stream_evaluate')
    def __init__(self, split: str, frame_fps: int, videos_pt_root_dir: str, annotations_root_dir: str, **kwargs):
        assert split in ['train', 'val', 'test']
        super().__init__(split=split, frame_fps=frame_fps, **kwargs)
        self.frame_fps = frame_fps

        seed = 42
        flipRng = random.Random(seed)

        originalFps = 30
        
        first_print = True #debug

        summarize_at_end = False #HardCoded

        shuffleDataset = True #HardCoded

        useRandFps = True #HARDCoded
        self.useRandFps = useRandFps

        remove_start = True #HardCoded

        boundary_learnFalse = True #HardCoded


        self.shiftViolenceStart_perc = 0.3
        # self.max_shiftViolenceStart_time = 3 #sec
       

        # violence_query = "Respond as soon as you detect a violence instance in the video" #NOTE should it have past desc or future desc about to punch or neither?
        add_query = True #HardCoded
        violence_query = "Analyze the given surveilance video and respond only when you detect an instance of violence in the streaming video, and respond with: 'Violence Detected!'. Do not respond if no violence is present."
        add_assistant_query_response = True #HardCoded
        assistant_violence_query_response = "Okay analyzing..."

        assistant_detectViolent = "Violence Detected!"

        user_summary_query = "Based on the preceding video frames, determine whether any violent behavior is present. Respond in exactly the following structure describing what you have seen. Violence Detected: [Yes/No]\nDescription: [Description of the actions in the video]\nAttacker: [Whether the left or right passenger is doing the violence, if any; otherwise 'None']\nCategory: [Interaction type]."

        anno_path = os.path.join(annotations_root_dir, f'vLLMonline_mola_{split}.json')

        print(f"Reading annotation from {anno_path}")
        annotations_json = json.load(open(anno_path))

        self.shuffleDataset = shuffleDataset
        if shuffleDataset:
            self.indices = list(range(len(annotations_json)))
            rng = random.Random(seed) 
            rng.shuffle(self.indices)

        self.rng_start_shift = random.Random(seed)

        annos = []
        for annotation in annotations_json:

            #annotation example:

            # annotation = {
            #     "videoName": "INCAR_20210427_Session_1_C4_P14_P13_1_rgb",
            #     "numFrames_nonViolent_segment": 320, #num frames before violence, in normal fps (30)
            #     #would be 0 if non violent video
            #     "numFrames_violent_segment": 200, #num frames before violence ends / video (if no extra segment, not from the 39 vids)
            #     "numFrames_nonViolent_extraSegment": 0, #default zero, only set if one of the 39 vids
            #     "numFrames_sampled": 32 #number of frames total sampled, found in .pt file

            #     "non_violent_seg_desc": "The left passenger and the right passenger are talking.",
            #     "non_violent_seg_label": "Verbal Interaction",
            #     "violent_seg_desc": "The left passenger pushes and then punches the right passenger.",
            #     "violent_seg_label": "Physical Assault",
            #     "summary": "The left passenger and the right passenger are talking when the left passenger pushes and then punches the right passenger.",
            #     "summary_flipped": "The right passenger and the left passenger are talking when the right passenger pushes and then punches the left passenger.",
            #     "summary_formatted": "Violence Detected: Yes\nDescription: The left passenger and the right passenger are talking when the left passenger pushes and then punches the right passenger.\nAttacker: The left passenger\nCategory: Physical assault",
            #     "summary_formatted_flipped": "Violence Detected: Yes\nDescription: The right passenger and the left passenger are talking when the right passenger pushes and then punches the left passenger.\nAttacker: The right passenger\nCategory: Physical assault"

            #     #will be added at the end of the video
            #     "user_summary_query" : "did you see any violence in the video?, answer in the following format ...", #could be the same format as iv2 chat?

            #     ###"assistant_summary_answer": "Violence Detected: yes\nDescription: The left and right are arguing then the right slaps the left\nAttacker: right\nCategory: physical assault"
            #     "rand_fps": 3 #if rand
            # }
            
            if useRandFps:
                frame_fps = annotation["rand_fps"]
                

            numFrames_nonViolent_segment_converted = self.convertNumFrames_toNewFps(
                numFrames=annotation["numFrames_nonviolent_segment"],
                originalFps=originalFps,
                newFps=frame_fps
                )
            # numFrames_violent_segment = self.convertNumFrames_toNewFps(
            #     numFrames=annotation["numFrames_violent_segment"],
            #     originalFps=self.originalFps,
            #     newFps=frame_fps
            #     )

            numFrames_restOfVideo = annotation["numFrames_sampled"] - numFrames_nonViolent_segment_converted

            conversation = [{'role': 'stream', 'num_frames': 1, 'learn': False},]

            if add_query:
                conversation += [{'role': 'user', 'content': violence_query}]
                if add_assistant_query_response:
                    conversation += [{'role': 'assistant', 'content': assistant_violence_query_response, 'learn': True}]

            
            if remove_start:
                isViolentVideo = annotation["numFrames_violent_segment"] > 0
                if isViolentVideo:
                    remove_start_numFrames = self.random_start_shift(nvSegmentLen = numFrames_nonViolent_segment_converted, isViolentVideo=isViolentVideo)
                else:
                    remove_start_numFrames = self.random_start_shift(nvSegmentLen = annotation["numFrames_sampled"], isViolentVideo=isViolentVideo)
            else:
                remove_start_numFrames = 0

            if annotation["numFrames_violent_segment"] > 0: #Violent video


                shiftViolenceStart = math.ceil(numFrames_restOfVideo * self.shiftViolenceStart_perc)
                # shiftViolenceStart = self.get_shiftViolenceStart(numFrames_restOfVideo, frame_fps)

                # print("\n")
                # print(f"VideoName = {annotation['videoName']}")
                # print(f"remove_start_numFrames = {remove_start_numFrames}, = {remove_start_numFrames / frame_fps} secs - @ {frame_fps} fps")
                # print(f"shiftViolenceStart = {shiftViolenceStart}")
                # print("\n")
                conversation_cont = [
                        {'role': 'stream', 'num_frames': numFrames_nonViolent_segment_converted - 1 - remove_start_numFrames + shiftViolenceStart, 'learn': True},
                        {'role': 'assistant', 'content': assistant_detectViolent, 'learn': True},
                        {'role': 'stream', 'num_frames': numFrames_restOfVideo - shiftViolenceStart, 'learn': True},
                        # {'role': 'assistant', 'content': f"The video related to the query \"{query}\" ends.", 'learn': True},
                    ]
                
                if boundary_learnFalse:
                    print(f"shiftViolenceStart = {shiftViolenceStart}")
                    conversation_cont[0]["learn"] = conversation_cont[0]["num_frames"] - shiftViolenceStart
                    conversation_cont[0]["force_addPred"] = True


                conversation += conversation_cont
                # if annotation["numFrames_nonViolent_extraSegment"] > 0:

                #     numFrames_nonViolent_extraSegment = self.convertNumFrames_toNewFps(
                #         numFrames=annotation["numFrames_nonViolent_extraSegment"],
                #         originalFps=self.originalFps,
                #         newFps=frame_fps
                #     )

                #     conversation.append({'role': 'stream', 'num_frames': numFrames_nonViolent_extraSegment, 'learn': True})



            else: #Non-violent video

                conversation += [
                        {'role': 'stream', 'num_frames': annotation["numFrames_sampled"] - 1 - remove_start_numFrames, 'learn': True},
                    ]
            

            flip = flipRng.choice([True, False])

            if summarize_at_end:

                # if role == 'user':
                #     fps_time = floor_time_by_fps(time, frame_fps, conversation[-1]['fps_time'], duration)
                #     if fps_time > duration:
                #         break
                #     if fps_time > conversation[-1]['fps_time']:
                #         conversation.append({'role': 'stream', 'num_frames': int((fps_time - conversation[-1]['fps_time']) * frame_fps), 'learn': True})
                #     conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})

                if flip:
                     summaryConv = [
                    {'role': 'user', 'content': user_summary_query}, # removed this part check if no problem > 'time': time, 'fps_time': fps_time <
                    {'role': 'assistant', 'content': annotation['summary_formatted_flipped'], 'learn': True},
                ]
                else:    
                    summaryConv = [
                        {'role': 'user', 'content': user_summary_query}, # removed this part check if no problem > 'time': time, 'fps_time': fps_time <
                        {'role': 'assistant', 'content': annotation['summary_formatted'], 'learn': True},
                    ]

                conversation.extend(summaryConv)
            else: #to fix token 60 issue , it needs a turn for chat template to add the \n after the ]
                    
                conversation[-1]["learn"] = conversation[-1]["num_frames"] - 1
                    


                
                
            
            if not conversation:
                continue

            


            if first_print:
                first_print = False
                print("debug conversation shape")
                for obj in conversation:
                    print(obj)
            
            videoName = annotation["videoName"]

            video_pt_path = os.path.join(videos_pt_root_dir,f"{videoName}.pt")
            if flip:
                if useRandFps:
                    video_pt_path = video_pt_path.replace("videos_sampled_randFps", "videos_sampled_randFps_flipped")
                else:
                    video_pt_path = video_pt_path.replace("videos_sampled", "videos_sampled_flipped")
                # print(f"Reading video from {video_pt_path}")
                # print(conversation)
            annos.append({
                'query': violence_query, #Violence Query
                'conversation': conversation,
                # 'load_ranges': {self.metadata[video_uid]['path']: range(int(video_start_time*frame_fps), int(last_time*frame_fps)+1)}

                'load_ranges': {video_pt_path: range(0 + remove_start_numFrames, annotation["numFrames_sampled"])}
                
            })
        self.annos = annos

    def compute_metrics(self, eval_predictions: EvalPrediction, *args, **kwargs):
        lm_ppl, frame_diff, fluency, lm_correctness = torch.from_numpy(eval_predictions.predictions).mean(dim=0).tolist()
        return {
            'lm_ppl': lm_ppl,
            'time_diff': frame_diff if self.useRandFps else frame_diff / self.frame_fps,  #return num frames diff if rand fps or time if static fps
            'fluency': fluency,
            'lm_correctness': lm_correctness,
        }

    def __len__(self):
        return len(self.annos)


    def convertNumFrames_toNewFps(self, numFrames, originalFps, newFps):
                new_numFrames = math.ceil((numFrames/originalFps) * newFps)
                return new_numFrames
    

    # def preprocess_conversation(self, conversation, query):
    #     query_prompt = random.choice(self.query_prompt_templates).replace('QUERY', query)
    #     return [{'role': 'user', 'content': query_prompt}] + conversation

    # def __getitem__(self, index):
    #     anno = self.annos[index]
    #     return *super().__getitem__(
    #         conversation=self.preprocess_conversation(anno['conversation'], anno['query']),
    #         load_ranges=anno['load_ranges'],
    #     ), index, self.evaluation_kwargs
    def __getitem__(self, index):
        if self.shuffleDataset:
            realIndex = self.indices[index]
        else:
            realIndex = index

        anno = self.annos[realIndex]

        # print("\nConversation:\n")
        # print(anno['conversation'])
        return *super().__getitem__(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs


    def random_start_shift(self, nvSegmentLen, isViolentVideo):
        nvSegmentLen -= 1 #because of 1 frame before query
        if isViolentVideo:
            shift_percent = self.rng_start_shift.uniform(0.0, 0.8)
            remove_start_numFrames = int(nvSegmentLen * shift_percent)

        else:
            firstPart = int(nvSegmentLen * 0.5)
            shift_percent = self.rng_start_shift.uniform(0.0, 0.5)
            remove_start_numFrames = int(firstPart * shift_percent)

        return remove_start_numFrames
    # def get_shiftViolenceStart(self, numFrames_restOfVideo, frame_fps):
    #     # shift_perc_v = math.ceil(numFrames_restOfVideo * self.shiftViolenceStart_perc)
    #     max_numFrames_shift_bySec = math.ceil(self.max_shiftViolenceStart_time * frame_fps)
    #     return min(numFrames_restOfVideo-frame_fps, max_numFrames_shift_bySec)




def build_ego4d_nlq_stream_mola_train_learnFalseBoundary(**kwargs):
    return NLQ_MOLA(split='train', **kwargs)

def build_ego4d_nlq_stream_mola_val_learnFalseBoundary(**kwargs):
    return NLQ_MOLA(split='val',**kwargs)

def build_ego4d_nlq_stream_mola_test_learnFalseBoundary(**kwargs):
    return NLQ_MOLA(split='test', **kwargs)


# do these when called in code, add to params
# NOTE: add system prompt
# NOTE: check  self.max_num_frames

#Augmentations add wrong text and makes it correct it so to reduce text dependency.. not needed now?

# if __name__ == '__main__':

#     system_prompt = (
#         "You are a vision-language model analyzing in-car surveillance video footage showing people seated "
#         "in the backseat of a vehicle. Respond only when you detect an instance of violence in the streaming "
#         "video, and respond with: 'Violence Detected!'. "
#         "Do not respond if no violence is present, unless the user explicitly asks a question."
#     )

#     videos_pt_root_dir = "/home/zeidan/Desktop/preprocess/videos_sampled_1+3x3_google--siglip-large-patch16-384/"


#     train_dataset = build_ego4d_nlq_stream_mola_train(
#         annotations_root_dir = "/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withSampledNumFrames_segDescSummary",
#         videos_pt_root_dir = videos_pt_root_dir, 
#         max_num_frames = float("inf"),
#         frame_fps=2, is_training=True, augmentation=False,
#         system_prompt=system_prompt, tokenizer=None,
#         vision_pretrained='google/siglip-large-patch16-384',
#         embed_mark='2fps_384_1+3x3'
#     )
#     val_dataset = build_ego4d_nlq_stream_mola_val(
#         annotations_root_dir = "/home/zeidan/Masters/videollm-online_fineTune_mola/annotating_mola/annotations_fromCombineMola_withSampledNumFrames_segDescSummary",
#         videos_pt_root_dir = videos_pt_root_dir,
#         max_num_frames = float("inf"),
#         frame_fps=2, is_training=True, augmentation=False,
#         system_prompt=system_prompt, tokenizer=None,
#         vision_pretrained='google/siglip-large-patch16-384',
#         embed_mark='2fps_384_1+3x3'
#     )
