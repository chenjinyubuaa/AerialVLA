import os
import json
import numpy as np
import copy
import cv2
from tqdm import tqdm

from llava.eval.aerialchat.utils import compute_iou, get_origin_action, crop_image_from_corners, move_view_corners

data_path = "./data/train_data.json"
image_path = "./data/train_images"
grid_size = 9
altitude_grid_size = 20

with open(data_path, "r") as f:
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
        
        pass
    
chat_instruction_data = []

sr = 0
for id, item in tqdm(enumerate(json_data)):
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
    
    # preprocess instructions
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
    
    chat_instruction_data_iter = []
    while t_index < 10:
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
            break
        if progress > 0.4:
            sr += 1
            chat_instruction_data.extend(chat_instruction_data_iter)
            break
        
        # im = cv2.imread(os.path.join(image_path, item['map_name'] + '.tif'), 1)
        # lng_ratio = item['lng_ratio']
        # lat_ratio = item['lat_ratio']
        # im_resized = cv2.resize(im, (int(im.shape[1] * lng_ratio / lat_ratio ), im.shape[0]), interpolation = cv2.INTER_AREA) # ratio_all = lat_ratio

        # crop_image, corner_info = crop_image_from_corners(
        #     img=im_resized,
        #     path_corners=copy.deepcopy(current_corner_list),
        #     directions=copy.deepcopy(current_direction_list),
        #     source=item,
        #     resize_shape=(384, 384)
        # )

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
        chat_instruction_data_iter.append(chat_instruction)
        
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
        t_index += 1
    
    if progress > 0.5 and t_index != 0:
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

sr = float(sr) / len(json_data)
print(f"sr : {sr}")

with open(f'train_aerial_instrucion_grid_{grid_size}_altitude_{altitude_grid_size}.json', 'w') as f:
    json.dump(chat_instruction_data, f, indent=4)
    
        
        
        
        
