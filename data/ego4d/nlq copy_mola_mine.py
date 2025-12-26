
import os, json, collections, torch, tqdm, random

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
    query_prompt_templates = [
        "Locate video clips related to the query \"QUERY\".",
        "Remind me when the query \"QUERY\".",
        "When query \"QUERY\" starts and ends, remind me.",
        "Do temporal grounding to query \"QUERY\".",
        "Can you locate query \"QUERY\" in the video?",
        "Record when query \"QUERY\".",
        "Please find the period of query \"QUERY\".",
        "Retrieve query \"QUERY\".",
        "Identify the start and end times of query \"QUERY\" in the video.",
        "Show me the video segment where query \"QUERY\" takes place.",
    ]
    evaluation_kwargs = DictWithTo(evaluator='stream_evaluate')
    def __init__(self, split: str, frame_fps: int, **kwargs):
        assert split in ['train', 'val', 'test']
        super().__init__(split=split, frame_fps=frame_fps, **kwargs)

        summarize_at_end = True #HardCoded



        anno_path = os.path.join(self.root, 'annotations', f'nlq_mola_{split}.json')

        annotations_json = json.load(open(anno_path))

        annos = []
        for annotation in annotations_json:

            #annotation example:

            # annotation = {
            #     "numFrames_nonViolent_segment": 320, #num frames before violence, in normal fps (30)
            #     #would be 0 if non violent video
            #     "numFrames_violent_segment": 200, #num frames before violence ends / video (if no extra segment, not from the 39 vids)
            #     "numFrames_nonViolent_extraSegment": 0, #default zero, only set if one of the 39 vids

            #     "assistant_answer_when_detectViolent":"I am detecting violence now.", #NOTE should it have past desc or future desc about to punch?

            #     "nonViolent_segment_description": "The left passenger and the right passenger are Arguing.",
            #     "violent_segment_description": "The right passenger slaps the left passenger",


            #     #will be added at the end of the video
            #     "user_summary_query" : "did you see any violence in the video, answer in the following format ...",
            #     #could be the same format as iv2 chat?
            #     "assistant_summary_answer": "Violence Detected: yes\nDescription: The left and right are arguing then the right slaps the left\nAttacker: right\nCategory: physical assault"

            # }

            numFrames_nonViolent_segment = self.convertNumFrames_toNewFps(
                numFrames=annotation["numFrames_nonViolent_segment"],
                originalFps=self.originalFps,
                newFps=frame_fps
                )
            numFrames_violent_segment = self.convertNumFrames_toNewFps(
                numFrames=annotation["numFrames_violent_segment"],
                originalFps=self.originalFps,
                newFps=frame_fps
                )
           

            if annotation["violent_segment_description"] > 0: #Violent video

                conversation = [
                        {'role': 'stream', 'num_frames': numFrames_nonViolent_segment, 'learn': True},
                        {'role': 'assistant', 'content': annotation['assistant_answer_when_detectViolent'], 'learn': True},
                        {'role': 'stream', 'num_frames': numFrames_violent_segment, 'learn': True},
                        # {'role': 'assistant', 'content': f"The video related to the query \"{query}\" ends.", 'learn': True},
                    ]
                if annotation["numFrames_nonViolent_extraSegment"] > 0:

                    numFrames_nonViolent_extraSegment = self.convertNumFrames_toNewFps(
                        numFrames=annotation["numFrames_nonViolent_extraSegment"],
                        originalFps=self.originalFps,
                        newFps=frame_fps
                    )

                    conversation.append({'role': 'stream', 'num_frames': numFrames_nonViolent_extraSegment, 'learn': True})



            else: #Non-violent video

                conversation = [
                        {'role': 'stream', 'num_frames': annotation["numFrames_nonViolent_segment"], 'learn': True},
                    ]
                
            if summarize_at_end:

                # if role == 'user':
                #     fps_time = floor_time_by_fps(time, frame_fps, conversation[-1]['fps_time'], duration)
                #     if fps_time > duration:
                #         break
                #     if fps_time > conversation[-1]['fps_time']:
                #         conversation.append({'role': 'stream', 'num_frames': int((fps_time - conversation[-1]['fps_time']) * frame_fps), 'learn': True})
                #     conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})

                summaryConv = [
                    {'role': 'user', 'content': annotation['user_summary_query']}, # removed this part check if no problem > 'time': time, 'fps_time': fps_time <
                    {'role': 'assistant', 'content': annotation['assistant_summary_answer'], 'learn': True},
                ]

                conversation.extend(summaryConv)
                
                
            
            if not conversation:
                continue
            annos.append({
                'query': query,
                'conversation': conversation,
                'load_ranges': {self.metadata[video_uid]['path']: range(int(video_start_time*frame_fps), int(last_time*frame_fps)+1)}
            })
        self.annos = annos


    def convertNumFrames_toNewFps(self, numFrames, originalFps, newFps):
                new_numFrames = int((numFrames/originalFps) * newFps)
                return new_numFrames

    def preprocess_conversation(self, conversation, query):
        query_prompt = random.choice(self.query_prompt_templates).replace('QUERY', query)
        return [{'role': 'user', 'content': query_prompt}] + conversation

    def __getitem__(self, index):
        anno = self.annos[index]
        return *super().__getitem__(
            conversation=self.preprocess_conversation(anno['conversation'], anno['query']),
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs





def build_ego4d_nlq_stream_train(**kwargs):
    return Ego4DStreamNLQ(split='train', **kwargs)

def build_ego4d_nlq_stream_val(**kwargs):
    return Ego4DStreamNLQ(split='val',**kwargs)

def build_ego4d_nlq_stream_test_unannotated(**kwargs):
    return Ego4DStreamNLQ(split='test', **kwargs)

if __name__ == '__main__':
    build_ego4d_nlq_stream_train(
        frame_fps=2, is_training=True, augmentation=True,
        system_prompt='', tokenizer=None,
        vision_pretrained='google/siglip-large-patch16-384',
        embed_mark='2fps_384_1+3x3'
    )
    build_ego4d_nlq_stream_val(
        frame_fps=2, is_training=True, augmentation=True,
        system_prompt='', tokenizer=None,
        vision_pretrained='google/siglip-large-patch16-384',
        embed_mark='2fps_384_1+3x3'
    )
