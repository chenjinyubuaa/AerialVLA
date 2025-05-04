import os
import json
import numpy as np
import copy
import cv2
from tqdm import tqdm
import torch
import math

from llava.eval.aerialchat.utils import compute_iou, get_origin_action, crop_image_from_corners, move_view_corners
from llava.eval.aerialchat.utils import get_map_info_dict, get_dialog_data

data_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/huitianrui/projects/project_2024/code/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json"
image_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/huitianrui/projects/project_2024/code/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images"
aug_data_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/code/LLaVA-NeXT/output/aerialchat/llava-onevision-qwen2-7b-ov-max1-lora-09-11-aerialchat-grid9-altitude-20-norm-sub-dagger/epoch_1/argument_datas'
grid_size = 9
altitude_grid_size = 20
origin_data_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/code/LLaVA-NeXT/train_aerial_instrucion_grid_9_altitude_20_sub.json"
output_data_path = f'aug_train_aerial_instrucion_grid_{grid_size}_altitude_{altitude_grid_size}.json'

def transforms_augment_data(
        data_path=data_path, 
        origin_data_path=origin_data_path,
        aug_data_path=aug_data_path, 
        output_data_path=output_data_path,
        grid_size=9,
        altitude_grid_size=20,
        aug_data_rate=1,
    ):
    with open(data_path, "r") as f:
        json_data = json.load(f)

    # map -> traj -> sub_traj
    map_dict = get_map_info_dict(json_data)
    full_dialog_data = get_dialog_data(map_dict)

    chat_instruction_data = []

    aug_data_files = os.listdir(aug_data_path)
    aug_data_list = []
    for file in aug_data_files:
        try:
            data = torch.load(os.path.join(aug_data_path, file))
            aug_data_list.extend(data)
        except:
            print(f"error load {file}")

    sr = 0
    for id, item in tqdm(enumerate(aug_data_list)):
        map_name = item['map_name']
        traj_id, sub_traj_id = item['route_index'].split('_')
        
        source_item = map_dict[map_name][traj_id][sub_traj_id]
        
        gt_path_corners = source_item['gt_path_corners']
        final_corner = gt_path_corners[-1]
        final_pos = np.mean(final_corner, axis=0)
        
        current_corner_list = item['current_corner_list']
        current_direction_list = item['current_direction_list']
        current_corner = np.array(current_corner_list[-1])
        current_pos = np.mean(current_corner, axis=0)
        current_direction = round(current_direction_list[-1]) % 360
        
        t_index = len(current_direction_list) - 1
        
        # move forward
        target = item['new_action']
        offset = target['offset']
        target['direction'] = (math.atan2(offset[0], offset[1]) / 3.14159 + 2) / 2 % 1
        target['distance'] = np.linalg.norm(offset) * (np.linalg.norm(current_corner[0] - current_corner[1]) / 2)
        target['altitude'] = target['altitude'].tolist()[0]
        # breakpoint()
        current_corner, current_direction = move_view_corners(
            current_corner,
            round(target['direction'] * 360),
            target['distance'],
            round(target['altitude'] * 360) + 40,
            source_item['gps_botm_left'],
            source_item['gps_top_right'],
            current_direction
        )
            
        current_corner_list.append(current_corner.tolist())
        current_direction_list.append(current_direction)

        progress = compute_iou(current_corner, final_corner)
        t_index += 1
        
        # process dialogs
        map_traj_index = f"{source_item['map_name']}_{source_item['route_index'].split('_')[0]}"
        sub_traj_index = int(source_item['route_index'].split('_')[1])
        dialog_data = copy.deepcopy(full_dialog_data[map_traj_index])
        
        conversation_template = []
        _, ans = dialog_data['instruction'].split('[QUE]')[-1].split('[INS]')
        conversation_template.append({"from": "human", "value": ans.strip()})

        for i, dialog in enumerate(dialog_data['dialogs'][:sub_traj_index-1]):
            que, ans = dialog.split('[QUE]')[-1].split('[INS]')
            conversation_template.append({"from": "gpt", "value": que.strip()})
            conversation_template.append({"from": "human", "value": ans.strip()})
            
        chat_instruction_data_iter = []
        try:
            target = get_origin_action(
                current_corner=current_corner,
                gt_path_corners=gt_path_corners,
                final_corner=final_corner,
                grid_size=grid_size,
                altitude_grid_size=altitude_grid_size,
            )
        except:
            print(f"error id: {id}")
            continue
        if (not (progress > 0.4)) and t_index < 10:

            altitude = int(round(target['altitude'], 2) * 100)
            x_offset = int(round(target['offset'][0], 2) * 50) + 50
            y_offset = int(round(target['offset'][1], 2) * 50) + 50
            iou_progress = int(round(target['progress'], 2) * 100)
            distance_progress = int(round(target['distance_progress'], 2) * 100)
            
            # format_response = "[action] " +  f"I will head to the {{<{x_offset}><{y_offset}>}} coordinates within my current view, set my altitude to <{altitude}>, and I believe the distance I have to the finish line is <{distance_progress}>%, and the IoU between my current view and the target area is <{iou_progress}>%."
            format_response = "[action] " + f"\noffset: {{<{x_offset}><{y_offset}>}}\naltitude: <{altitude}>\ndistance_progress:<{distance_progress}>%\niou_progress:<{iou_progress}>%"
            conversation = copy.deepcopy(conversation_template)
            conversation.append({"from": "gpt", "value": format_response})
            
            chat_instruction = dict(
                map_name=source_item['map_name'],
                route_index=source_item['route_index'],
                conversations=conversation,
                t_index=t_index,
                meta=dict(
                    gps_botm_left=source_item['gps_botm_left'],
                    gps_top_right=source_item['gps_top_right'],
                    lng_ratio=source_item['lng_ratio'],
                    lat_ratio=source_item['lat_ratio'],
                    current_corner_list=copy.deepcopy(current_corner_list),
                    current_direction_list=copy.deepcopy(current_direction_list),
                    target=target,
                )
            )
            chat_instruction_data_iter.append(chat_instruction)
            
            current_corner, current_direction = move_view_corners(
                current_corner,
                round(target['direction'] * 360),
                target['distance'],
                round(target['altitude'] * 360) + 40,
                source_item['gps_botm_left'],
                source_item['gps_top_right'],
                current_direction
            )
            
            current_corner_list.append(current_corner.tolist())
            current_direction_list.append(current_direction)

            progress = compute_iou(current_corner, final_corner)
            t_index += 1
            
        chat_instruction_data.extend(chat_instruction_data_iter)
        
        if progress > 0.4 and t_index != 0:
            try:
                iou_progress = int(round(target['progress'], 2) * 100)
            except:
                target = get_origin_action(
                    current_corner=current_corner,
                    gt_path_corners=gt_path_corners,
                    final_corner=final_corner,
                    grid_size=grid_size,
                    altitude_grid_size=altitude_grid_size,
                )
                iou_progress = int(round(target['progress'], 2) * 100)
            distance_progress = int(round(target['distance_progress'], 2) * 100)
            progress_format_response = f"I believe the distance I have to the finish line is <{distance_progress}>%, the IoU between my current view and the target area is <{iou_progress}>% and I need to ask questions regarding my next action."
            
            if sub_traj_index != dialog_data["sub_traj_num"]:
                response = dialog_data['dialogs'][sub_traj_index-1]
            else:
                response = "[QUE] Am I near the destination? How to go to destinaton? [INS] You have arrived."
            que, ans = response.split('[QUE]')[-1].split('[INS]')
            
            format_response = "[question] " + progress_format_response + " " + que.strip()
            conversation = copy.deepcopy(conversation_template)
            conversation.append({"from": "gpt", "value": format_response})
            
            chat_instruction = dict(
                map_name=item['map_name'],
                route_index=item['route_index'],
                conversations=conversation,
                t_index=t_index,
                meta=dict(
                    gps_botm_left=item['gps_botm_left'],
                    gps_top_right=item['gps_top_right'],
                    lng_ratio=item['lng_ratio'],
                    lat_ratio=item['lat_ratio'],
                    current_corner_list=copy.deepcopy(current_corner_list),
                    current_direction_list=copy.deepcopy(current_direction_list),
                    target=target,
                )
            )
            chat_instruction_data.append(chat_instruction)
        
    sr = float(sr) / len(aug_data_list)
    # print(f"sr : {sr:.4f}")
    print(f"new traj num : {len(chat_instruction_data)}")

   
    with open(origin_data_path, 'r') as f:
        origin_data = json.load(f)
    print(f"origin traj num: {len(origin_data)}")
    # augment data rate
    import random
    random.shuffle(chat_instruction_data)
    chat_instruction_data = chat_instruction_data[:int(len(origin_data) * aug_data_rate)]
    chat_instruction_data.extend(origin_data)
    print(f'new full traj num: {len(chat_instruction_data)}')
    
    with open(output_data_path, 'w') as f:
        json.dump(chat_instruction_data, f, indent=4)
        
if __name__ == '__main__':
    transforms_augment_data(
        data_path=data_path,
        aug_data_path=aug_data_path,
        grid_size=grid_size,
        altitude_grid_size=altitude_grid_size,
    )