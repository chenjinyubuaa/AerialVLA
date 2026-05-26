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

from llava.eval.aerialchat.utils import compute_iou, get_origin_action, crop_image_from_corners, move_view_corners, get_corner_prompt, eval_metrics
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

from llava.model.builder import load_lora
from llava.utils import rank0_print

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

def eval_model(args, model=None, tokenizer=None, image_processor=None):
    
    # Model
    if model is None:
        disable_torch_init()
        tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_base, None, "aerialchat_llava_qwen")  # Add any other thing you want to pass in llava_model_args

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

    # Data
    with open(args.eval_data_path, "r") as f:
        json_data = json.load(f)
    # map -> traj -> sub_traj
    map_dict = dict()
    for _, item in enumerate(json_data):
        map_name = item['map_name']
        traj_id, sub_traj_id = item['route_index'].split('_')
        
        if map_name not in map_dict:
            map_dict[map_name] = dict()
        if traj_id not in map_dict[map_name]:
            map_dict[map_name][traj_id] = dict()
        
        item['angle'] = round(item['angle']) % 360
        item['instructions'] = item['instructions']
        item['gt_path_corners'] = np.array(item['gt_path_corners'])
        
        map_dict[map_name][traj_id][sub_traj_id] = item 
        
    full_dialog_data = dict()
    for map_name, map_data in map_dict.items():
        for traj_id, traj_data in map_data.items():
            
            map_traj_index = f"{map_name}_{traj_id}"
            sub_traj_num = traj_data['1']['last_round_idx']
            
            map_traj_data = {
                'instruction': traj_data['1']['instructions'],
                'dialogs': [],
                'sub_traj_num': sub_traj_num,
                
            }
            for sub_traj_id in range(2, sub_traj_num+1):
                map_traj_data['dialogs'].append(traj_data[str(sub_traj_id)]['instructions'])
            
            full_dialog_data[map_traj_index] = map_traj_data

    image_folder = args.image_folder 
    correct_traj = 0
    all_traj_info = dict()

    json_data = get_chunk(json_data, args.num_chunks, args.chunk_idx)
    for id, item in tqdm(enumerate(json_data)):
        traj_info = dict(
            instr_id=f"{item['map_name']}__{item['route_index']}",
            gt_progress=[],
            gt_path_corners=item['gt_path_corners'],
            path_corners=[],
        )
        
        gt_path_corners = item['gt_path_corners']
        final_corner = gt_path_corners[-1]
        final_pos = np.mean(final_corner, axis=0)
        current_corner = gt_path_corners[0]
        current_pos = np.mean(current_corner, axis=0)
        current_direction = round(item['angle']) % 360
        
        current_corner_list = [current_corner.tolist()]
        current_direction_list = [current_direction]
        
        t_index = 0
        progress = compute_iou(current_corner, final_corner)
        
        map_traj_index = f"{item['map_name']}_{item['route_index'].split('_')[0]}"
        sub_traj_index = int(item['route_index'].split('_')[1])
        dialog_data = copy.deepcopy(full_dialog_data[map_traj_index])
            
        conversation_template = []
        _, ans = dialog_data['instruction'].split('[QUE]')[-1].split('[INS]')
        conversation_template.append({"from": "human", "value": ans.strip()})

        for i, dialog in enumerate(dialog_data['dialogs'][:sub_traj_index-1]):
            que, ans = dialog.split('[QUE]')[-1].split('[INS]')
            conversation_template.append({"from": "gpt", "value": que.strip()})
            conversation_template.append({"from": "human", "value": ans.strip()})
        conversation_template.append({"from": "gpt", "value": "[action]"})
        
        image_file = item["map_name"] + ".tif"
        image = cv2.imread(os.path.join(image_folder, image_file), 1)
        image = cv2.resize(image, (int(image.shape[1]*item["lng_ratio"]/item["lat_ratio"]), image.shape[0]), interpolation=cv2.INTER_AREA)
        image = image[:,:,::-1] # convert to RGB
        
        from llava.eval.aerialchat.utils import crop_image_from_corners, crop_corner_for_map, postprocess_pred_res
        
        while t_index < 10:
            history_map_image, corner_info = crop_image_from_corners(image, current_corner_list, current_direction_list, item, resize_shape=None)
            current_image = crop_corner_for_map(
                corner=current_corner_list[-1], 
                image_map=image, 
                width=384,
                height=384,
                source=item,
            )
            
            conversation = copy.deepcopy(conversation_template)
            system_message = aerialchat_system
            corner_prompt = get_corner_prompt(corner_info)
            system_message = system_message.replace("<corners>", corner_prompt)
            input_ids = preprocess_qwen(conversation, tokenizer, has_image=True,system_message=system_message)
            input_ids = input_ids.cuda()

            images = [Image.fromarray(history_map_image), Image.fromarray(current_image)]
            image_sizes = [image.size for image in images]
            image_tensor = process_images(images, image_processor, model.config)
            image_tensor_copy = [image_tensor[0], image_tensor[1][0]]
            image_tensor = image_tensor_copy
            image_tensor = [_image.bfloat16().cuda() for _image in image_tensor]

            with torch.inference_mode():
                output = model(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes)
                # output_ids = model.generate(
                #     input_ids,
                #     images=image_tensor,
                #     image_sizes=image_sizes,
                #     do_sample=True if args.temperature > 0 else False,
                #     temperature=args.temperature,
                #     top_p=args.top_p,
                #     num_beams=args.num_beams,
                #     # no_repeat_ngram_size=3,
                #     max_new_tokens=1024,
                #     use_cache=True)
            
            # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            # outputs = outputs.strip()
            
            # pattern = r"offset: \{<(\S+)><(\S+)>\}\naltitude: <(\S+)>\ndistance_progress:<(\S+)>%\niou_progress:<(\S+)>%"
            # match = re.search(pattern, outputs)
            # if match:
            
            pred_target = output['pred_actions']
            target_list = postprocess_pred_res(pred_target['offset'], pred_target['altitude'], pred_target['iou_progress'], pred_target['distance_progress'])
            
            target = target_list[0]
            # target['offset'] = target['grid']
            target['direction'] = (math.atan2(target["offset"][0], target["offset"][1]) /3.14159 + 2) / 2 % 1
            target['distance'] = np.linalg.norm(target["offset"]) * (np.linalg.norm(current_corner[0] - current_corner[1])/2)
        
            current_corner, current_direction = move_view_corners(
                current_corner,
                round(target['direction'] * 360),
                target['distance'],
                round(target['altitude'] * 360) + 40,
                item['gps_botm_left'],
                item['gps_top_right'],
                current_direction
            )
            
            current_corner_list.append(current_corner.tolist())
            current_direction_list.append(current_direction)

            progress = compute_iou(current_corner, final_corner)
            traj_info['gt_progress'].append(progress)
            t_index += 1
            
            if target['progress'] > 0.4:
                break
    
        traj_info['path_corners'] = current_corner_list
        all_traj_info[traj_info['instr_id']] = traj_info
     
        if progress > 0.4:
            correct_traj += 1
    print(f"naive sr: {correct_traj/len(json_data)}")
    
    dataset_name = args.eval_data_path.split('/')[-1]
    if 'test' not in args.eval_data_path:
        avg_metrics, metrics = eval_metrics(all_traj_info)
        log = ""
        for k, v in avg_metrics.items():
            v = round(v, 2)
            log += f"{k}: {v} "
        print(f"{dataset_name} result: {log}")
        return avg_metrics
    else:
        for k in all_traj_info:
            all_traj_info[k] = {
                'path_corners': all_traj_info[k]['path_corners']
            }
        import datetime
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        file_dir = os.path.join(args.output_dir, 'submit', f'{dataset_name}_{time}')
        os.makedirs(os.path.dirname(file_dir), exist_ok=True)
        np.save(file_dir, all_traj_info)

    
            
            
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default="./models/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-path-stage1", type=str, default="./checkpoints/geochat-stage2")
    parser.add_argument("--model-path-stage2", type=str, default="output/aerialchat/llava-onevision-qwen2-7b-si-max9-lora-09-07-aerial-chat-re")

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-data-path", type=str, default="./data/train_1094.json")
    parser.add_argument("--image-folder", type=str, default="./data/train_images")
    
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