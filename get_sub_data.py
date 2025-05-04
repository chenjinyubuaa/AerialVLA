import json

data_name = 'train_aerial_instrucion_grid_9_altitude_20'

full_data = f'/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/code/LLaVA-NeXT/{data_name}.json'
output_data = f'/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/code/LLaVA-NeXT/{data_name}_sub.json'

full_data_list = json.load(open(full_data, 'r'))

print(f'full length: {len(full_data_list)}')

sub_data_list = [data for data in full_data_list if data['map_name'] == '1094']

print(f'sub length: {len(sub_data_list)}')
json.dump(sub_data_list, open(output_data, 'w'), indent=4)