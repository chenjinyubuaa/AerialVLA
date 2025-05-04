import argparse
import math
import json
import numpy as np
from tqdm import tqdm
import copy
import cv2
import os
from PIL import Image
import torch
import re

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List

from llava.eval.aerialchat.utils import compute_iou, get_origin_action, crop_image_from_corners, move_view_corners, get_corner_prompt
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.eval.aerialchat.utils import crop_image_from_corners, crop_corner_for_map, postprocess_pred_res

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

aerialchat_system = "You are an expert drone pilot, capable of effectively executing flight missions based on instructions, outputting decision content, and asking questions to the navigator at appropriate moments. You are now provided with the historical map of the remote sensing views you have flown over <image>, the coordinates of observed areas according to each step of your historical decisions on the map: <corners>, and the current area image you are observing <image>. Please provide the next step decision based on the instructions."

from llava.eval.model_vqa import preprocess_qwen

from llava.model import LlavaQwenForCausalLM

def eval_model(args):
    with open(args.data_path, "r") as f:
        json_data = json.load(f)

    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_base, None, "aerialchat_llava_qwen")  # Add any other thing you want to pass in llava_model_args


    from llava.model.builder import load_lora
    from llava.utils import rank0_print

    model = load_lora(model, args.model_path_stage1)
    rank0_print("Merging LoRA weights...")
    model = model.merge_and_unload()
    rank0_print("Model is loaded...")
    model = load_lora(model, args.model_path_stage2)
    rank0_print("Merging LoRA weights...")
    model = model.merge_and_unload()
    rank0_print("Model is loaded...")
    
    model.initialize_added_tokenizer(tokenizer)
    model.config.image_aspect_ratio = "anyres_max_1"
    
    image_folder = args.image_folder 
    correct_traj = 0
    losses = 0
    for id, item in tqdm(enumerate(json_data)):
        meta = item['meta']
        
        image_file = item["map_name"] + ".tif"
        image = cv2.imread(os.path.join(image_folder, image_file), 1)
        image = cv2.resize(image, (int(image.shape[1]*meta["lng_ratio"]/meta["lat_ratio"]), image.shape[0]), interpolation=cv2.INTER_AREA)
        image = image[:,:,::-1] # convert to RGB
        
        history_map_image, corner_info = crop_image_from_corners(
            image, 
            meta['current_corner_list'], 
            meta['current_direction_list'], 
            meta, 
            resize_shape=None
        )
        current_image = crop_corner_for_map(
            corner=meta['current_corner_list'][-1], 
            image_map=image, 
            width=384,
            height=384,
            source=meta,
        )
        images = [Image.fromarray(history_map_image), Image.fromarray(current_image)]
        image_sizes = [image.size for image in images]
        image_tensor = process_images(images, image_processor, model.config)
        image_tensor_copy = [image_tensor[0], image_tensor[1][0]]
        image_tensor = image_tensor_copy
        image_tensor = [_image.half().cuda() for _image in image_tensor]
        
        sources = copy.deepcopy([e["conversations"] for e in [item]])
        system_message = aerialchat_system
        corner_prompt = get_corner_prompt(corner_info)
        system_message = system_message.replace("<corners>", corner_prompt)
        input_ids = preprocess_qwen(sources[0], tokenizer, has_image=True, system_message=system_message)
        input_ids = input_ids.cuda()
        
        current_corner = np.array(meta['current_corner_list'][-1])
        current_direction = meta['current_direction_list'][-1]

        with torch.inference_mode():
            output = model(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                metas=[meta]
            )
            gt_target = meta['target']
            if 'grid' in meta['target']:
                grid_size = 9
                _index = meta['target']['grid'][1] * grid_size + meta['target']['grid'][0]
                one_hot = torch.zeros(grid_size * grid_size)
                one_hot[_index] = 1
                gt_target['grid_onehot'] = one_hot
                
            for k in gt_target:
                gt_target[k] = torch.tensor(gt_target[k]).unsqueeze(0)
                gt_target[k] = gt_target[k].cuda()
            
            
            
            # output['pred_actions']['offset'] = output['pred_actions']['offset'] * 2 - 1
            loss = model.compute_action_loss(output['pred_actions'], meta)
            print(loss)
            losses += loss
            pred_target = output['pred_actions']
            pass
            # target_list = postprocess_pred_res(pred_target['offset'], pred_target['altitude'], pred_target['iou_progress'], pred_target['distance_progress'])
            
            # target = target_list[0]
            # # target['offset'] = target['offset'] * 2 - 1
            # target['direction'] = (math.atan2(target["offset"][0], target["offset"][1]) /3.14159 + 2) / 2 % 1
            # target['distance'] = np.linalg.norm(target["offset"]) * (np.linalg.norm(current_corner[0] - current_corner[1])/2)
            
            # current_corner, current_direction = move_view_corners(
            #     current_corner,
            #     round(target['direction'] * 360),
            #     target['distance'],
            #     round(target['altitude'] * 360) + 40,
            #     meta['gps_botm_left'],
            #     meta['gps_top_right'],
            #     current_direction
            # )

            # progress = compute_iou(current_corner, final_corner)
            # t_index += 1
            
    #         if target['progress'] > 0.5:
    #             break

    #     if progress > 0.5:
    #         correct_traj += 1
    # print(f"sr: {correct_traj/len(json_data)}")
            
    print(losses/len(json_data))
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default="/mnt/data3/cjy/AerialChat-Llava-OV-main/llava-onevision-qwen2-0.5b-ov")
    parser.add_argument("--model-path-stage1", type=str, default="/mnt/data3/cjy/AerialChat-Llava-OV/output/debug-grid27")
    parser.add_argument("--model-path-stage2", type=str, default="output/aerialchat/llava-onevision-qwen2-7b-si-max9-lora-09-07-aerial-chat-re")

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="/mnt/data3/cjy/AerialChat-Llava-OV/sub_aerial_data/train_aerial_instrucion_grid_sub.json")
    parser.add_argument("--image-folder", type=str, default="/mnt/data3/cjy/AerialChat-Llava-OV/sub_aerial_data/")
    
    parser.add_argument("--max_action_len",type=int, default=10)
    parser.add_argument("--batch_size",type=int, default=1)
    
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    args = parser.parse_args()

    args.output_dir = args.model_path_stage2
    
    eval_model(args)