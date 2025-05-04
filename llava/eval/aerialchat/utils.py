import json
import os
import math
import numpy as np
import random
from collections import defaultdict
import cv2 
import torch
import shapely
import shapely.geometry
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import nearest_points

def gps_to_img_coords(gps, ob):
    gps_botm_left = ob['gps_botm_left']
    gps_top_right = ob['gps_top_right']
    lng_ratio = ob['lng_ratio']
    lat_ratio = ob['lat_ratio']
    
    return int(round((gps[1] - gps_botm_left[1]) / lat_ratio)), int(round((gps_top_right[0] - gps[0]) / lat_ratio))

def corner_gps_to_img(gps_corner, ob):
    img_corner = gps_corner
    for i in range(gps_corner.shape[0]):
        img_corner[i] = gps_to_img_coords(gps_corner[i], ob)
    return img_corner

def create_corners_mask(image_shape, corners):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # 创建一个与图像大小相同的空掩码

    for corner in corners:
        pts = np.array(corner, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)  # 在掩码上绘制白色多边形

    return mask

def apply_mask(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def crop_to_bounding_box(image, mask):
    x, y, w, h = cv2.boundingRect(mask)
    side_length = max(w, h)  # 取宽度和高度中的较大值作为正方形的边长

    # 调整矩形的位置和大小以确保其仍然包含所有的有效内容
    center_x = x + w // 2
    center_y = y + h // 2

    new_x = max(center_x - side_length // 2, 0)
    new_y = max(center_y - side_length // 2, 0)

    new_x = min(new_x, image.shape[1] - side_length)
    new_y = min(new_y, image.shape[0] - side_length)

    cropped_image = image[new_y:new_y + side_length, new_x:new_x + side_length]
    bounding_box = (new_x, new_y, side_length)

    return cropped_image, bounding_box

def convert_coordinates(original_coords, bounding_box, original_size, target_size):
    """
    将原图上的坐标点转换到裁剪且调整大小后的图像上的坐标。

    参数:
    - original_coords: 原图上的坐标点列表 [(x1, y1), (x2, y2), ...]
    - bounding_box: 裁剪的外接矩形 (x, y, side_length)
    - original_size: 原图的大小 (width, height)
    - target_size: 目标图像的大小 (width, height)

    返回:
    - new_coords: 转换后的坐标点列表 [(x1, y1), (x2, y2), ...]
    """
    new_coords = []
    x_offset, y_offset, side_length = bounding_box
    scale_x = target_size[0] / side_length
    scale_y = target_size[1] / side_length

    for (x, y) in original_coords:
        new_x = (x - x_offset) * scale_x
        new_y = (y - y_offset) * scale_y
        new_coords.append((int(new_x), int(new_y)))

    return new_coords

def crop_corner_for_map(corner, image_map, width=224, height=224, is_saliency=False, source=dict()):
    view_area_corner = np.array(corner)

    # generate view area
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    img_coord_view_area_corner = view_area_corner
    for xx in range(img_coord_view_area_corner.shape[0]):
        img_coord_view_area_corner[xx] = gps_to_img_coords(view_area_corner[xx], source)
    img_coord_view_area_corner = np.array(img_coord_view_area_corner, dtype="float32")

    M = cv2.getPerspectiveTransform(img_coord_view_area_corner, dst_pts)

    im_view = cv2.warpPerspective(image_map, M, (width, height))

    if is_saliency:
        im_view = np.asarray(cv2.cvtColor(im_view, cv2.COLOR_BGR2GRAY)) / 255

    return im_view


def get_obs_for_map(source, corner=None, directions=None, t=None, 
                    shortest_teacher=False, width=224, height=224):
    if t == None:
        t_input = 0
    else:
        if t < len(source['gt_path_corners']):
            t_input = t
        else:
            t_input = len(source['gt_path_corners']) - 1
    
    if corner is None:
        view_area_corner = source['gt_path_corners'][t_input]
    else:
        view_area_corner = corner

    # generate view area
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    view_area_corner = np.array(view_area_corner)
    img_coord_view_area_corner = view_area_corner
    for xx in range(img_coord_view_area_corner.shape[0]):
        img_coord_view_area_corner[xx] = gps_to_img_coords(view_area_corner[xx], source)
    img_coord_view_area_corner = np.array(img_coord_view_area_corner, dtype="float32")
    
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(img_coord_view_area_corner, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    im_view = cv2.warpPerspective(source['image_map'], M, (width, height))
    gt_saliency = cv2.warpPerspective(source['attention_map'], M, (width, height))
    gt_saliency = np.asarray(cv2.cvtColor(gt_saliency, cv2.COLOR_BGR2GRAY)) / 255
    
    
    return {
        'current_view': im_view, 
        'gt_saliency': gt_saliency, 
        'view_area_corners': view_area_corner,

        'gps_botm_left': source['gps_botm_left'],
        'gps_top_right': source['gps_top_right'],
        'lng_ratio': source['lng_ratio'],
        'lat_ratio': source['lat_ratio'],
        'starting_angle': source['starting_angle'],
        'gt_path_corners': source['gt_path_corners'],
    }

def get_obs_for_map_batch(sources, corners=None, t=None, width=224, height=224):
    obs_list = []
    for i, source in enumerate(sources):
        corner = None if corners is None else corners[i]
        obs = get_obs_for_map(source, corner, t=t, width=width, height=height)
        obs_list.append(obs)
    return obs_list

def move_view_corner_forward(cs, change): # corners => cs
    new_cs = np.zeros((4,2))
    new_cs[0] = cs[0] + (cs[0] - cs[3])/ np.linalg.norm((cs[3] - cs[0])) * change
    new_cs[1] = cs[1] + (cs[1] - cs[2])/ np.linalg.norm((cs[2] - cs[1])) * change
    new_cs[2] = cs[2] + (cs[1] - cs[2])/ np.linalg.norm((cs[2] - cs[1])) * change
    new_cs[3] = cs[3] + (cs[0] - cs[3])/ np.linalg.norm((cs[3] - cs[0])) * change
    return new_cs

def rotation_anticlock(theta, p):
    M = np.array([
        [np.cos(theta / 180 * 3.14159), np.sin(theta / 180 * 3.14159)], 
        [-np.sin(theta / 180 * 3.14159), np.cos(theta / 180 * 3.14159)]
    ])
    return np.matmul(M, np.array([p[0], p[1]]))

def change_corner(cs, change): # corners = cs
    new_cs = np.zeros((4,2))
    new_cs[0] = cs[0] + (cs[0] - cs[1])/ np.linalg.norm((cs[1] - cs[0])) * change
    new_cs[0] += (cs[0] - cs[3])/ np.linalg.norm((cs[3] - cs[0])) * change

    new_cs[1] = cs[1] + (cs[1] - cs[0])/ np.linalg.norm((cs[1] - cs[0])) * change
    new_cs[1] += (cs[1] - cs[2])/ np.linalg.norm((cs[2] - cs[1])) * change

    new_cs[2] = cs[2] + (cs[2] - cs[3])/ np.linalg.norm((cs[2] - cs[3])) * change
    new_cs[2] += (cs[2] - cs[1])/ np.linalg.norm((cs[2] - cs[1])) * change

    new_cs[3] = cs[3] + (cs[3] - cs[2])/ np.linalg.norm((cs[2] - cs[3])) * change
    new_cs[3] += (cs[3] - cs[0])/ np.linalg.norm((cs[3] - cs[0])) * change

    return new_cs

def get_direction(start, end):
    vec=np.array(end) - np.array(start)
    _angle = 0
    #          90
    #      135    45
    #     180  .    0
    #      225   -45 
    #          270
    if vec[1] > 0: # lng is postive
        _angle = np.arctan(vec[0]/vec[1]) / 1.57*90
    elif vec[1] < 0:
        _angle = np.arctan(vec[0]/vec[1]) / 1.57*90 + 180
    else:
        if np.sign(vec[0]) == 1:
            _angle = 90
        else:
            _angle = 270
    _angle = (360 - _angle+90)%360
    return _angle

def move_view_corners(corners, angle, distance, altitude, gps_botm_left, gps_top_right, input_current_direction = None): 
    '''
    rotate first and then move forward;
    return the unchanged if hit the map edge
    '''
    current_direction = round(get_direction(np.mean(corners, axis=0),(corners[0] + corners[1])/2)) % 360
    if input_current_direction != None and abs(input_current_direction - current_direction) >2:
        print('warning, currencting the view area by: +', input_current_direction - current_direction)
        angle += input_current_direction        
    # -------- Zoom --------
    current_view_area_edge_length = np.linalg.norm((corners[1]) - corners[0])*11.13*1e4
    # print('step_to_zoom: ',altitude*400, current_view_area_edge_length)
    # print(corners)
    step_change_of_view_zoom = 0.5*(altitude - current_view_area_edge_length)/11.13/1e4
    _new_corners = change_corner(
        corners,
        step_change_of_view_zoom
    )
    corners = _new_corners

    # -------- Rotate --------

    # print(angle)
    mean_im_coords = np.mean(corners, axis=0)
    _corners = [
        corners[0] - mean_im_coords,
        corners[1] - mean_im_coords,
        corners[2] - mean_im_coords,
        corners[3] - mean_im_coords
    ]  # counter clock wise

    rotated_corners = []
    for i in range(4):
        rotated_point = mean_im_coords + rotation_anticlock(-angle, _corners[i])
        rotated_corners.append(rotated_point)

    # -------- Move --------

    
    step_change_of_view_move = distance
    _new_corners = move_view_corner_forward(
            np.array(rotated_corners),
            step_change_of_view_move)


    new_corners = []
    for i in _new_corners:

        if i[0]>gps_botm_left[0] and i[0] < gps_top_right[0] and i[1]>gps_botm_left[1] and i[1]<gps_top_right[1]:
            new_corners.append(i)
        else:
            break
    if len(new_corners) != 4:
        return np.array(rotated_corners), (current_direction + angle) % 360
    else:
        return np.array(new_corners), (current_direction + angle) % 360

def compute_iou(a, b):
    a = np.array(a)  # quadrilateral two-dimensional coordinate representation
    poly1 = Polygon(
        a).convex_hull  # python quadrilateral object, will automatically calculate four points, the last four points in the order of: top left bottom right bottom right top left top
    # print(Polygon(a).convex_hull)  # you can print to see if this is the case

    b = np.array(b)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # Merge two box coordinates to become 8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # contains the smallest polygon point of the two quadrilaterals
    if not poly1.intersects(poly2):  # If the two quadrilaterals do not intersect
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # intersection area
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area)/(union_area-inter_area)  #wrong
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # The source code gives two ways to calculate IOU, the first one is: intersection part / area of the smallest polygon containing two quadrilaterals
            # The second one: intersection/merge (common way to calculate IOU of rectangular box)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def teacher_action(obs, ended, corners, directions, feedback='teacher'):
    """
    Extract teacher actions into variable.
    :param obs: The observation.
    :param ended: Whether the action seq is ended
    :return:
    """

    teacher_a = [['0','0'] for x in range(len(obs))]
    progress = np.zeros((len(obs),1), dtype=np.float32)
    for i in range(len(obs)):
        
        current_pos = np.mean(corners[i], axis = 0)

        # -------- calculate the progress (iou) --------   
        iou = compute_iou(corners[i], obs[i]['gt_path_corners'][-1])

        progress[i] = np.float32(iou)
        
        # -------- find teacher altitude --------
        min_dis = 1000
        for j in range(len(obs[i]['gt_path_corners'])-1, -1, -1):
            gt_pos = np.mean(obs[i]['gt_path_corners'][j], axis = 0)
            dis_to_current = np.linalg.norm(gt_pos - current_pos)
            if dis_to_current+0.00001<min_dis: # 0.00001 is in case there are two gt_path_corner are the same
                min_dis = dis_to_current
                closest_step_index = j
        teacher_a[i][1] = float((np.linalg.norm(obs[i]['gt_path_corners'][closest_step_index][0] -\
                            obs[i]['gt_path_corners'][closest_step_index][1])\
                            *11.13*1e4 -40 )/ (400-40))
        if ended[i] or progress[i] > 0.5:
            teacher_a[i][0] = np.array([0,0], dtype=np.float32)
            continue                   

        # -------- find teacher next_pos --------        
        goal_corner_center = np.mean(obs[i]['gt_path_corners'][-1], axis = 0)
        polygon = corners[i]
        shapely_poly = shapely.geometry.Polygon(polygon)
        # in teacher forcing learning, the trajectory will be followed step by step
        if feedback == 'student': 
            target_point_index = -1
            line = [current_pos]+[np.mean(obs[i]['gt_path_corners'][target_point_index], axis = 0)]
            shapely_line = shapely.geometry.LineString(line)
            intersection_line = list(shapely_poly.intersection(shapely_line).coords)
        else:
            line = [np.mean(obs[i]['gt_path_corners'][j], axis = 0) for j in range(len(obs[i]['gt_path_corners']))]
            shapely_line = shapely.geometry.LineString(line)
            if type(shapely_poly.intersection(shapely_line)) == shapely.geometry.linestring.LineString:
                intersection_line = list(shapely_poly.intersection(shapely_line).coords)
            else:
                intersection_line = []
                for line_string in shapely_poly.intersection(shapely_line):
                    intersection_line += list(line_string.coords)
        

            if intersection_line == []:
                print(line, closest_step_index)
                target_point_index = -1
                line = [current_pos]+[np.mean(obs[i]['gt_path_corners'][target_point_index], axis = 0)]
                shapely_line = shapely.geometry.LineString(line)
                intersection_line = list(shapely_poly.intersection(shapely_line).coords)
            
            
        if intersection_line == []:
            print(line, closest_step_index)
        
        min_distance = 1
        for x in intersection_line:
            x = np.array(x)
            _distance = np.linalg.norm(x- goal_corner_center) 
            if _distance < min_distance:
                min_distance = _distance
                teacher_a[i][0] = x

        _net_next_pos = 1e5*(teacher_a[i][0] - current_pos)
        _net_y = np.round(1e5*((corners[i][0] + corners[i][1])/2 - current_pos)).astype(int)
        _net_x = np.round(1e5*((corners[i][1] + corners[i][2])/2 - current_pos)).astype(int)

        A = np.mat([[_net_x[0],_net_y[0]],[_net_x[1],_net_y[1]]])
        b = np.mat([_net_next_pos[0],_net_next_pos[1]]).T
        r = np.linalg.solve(A,b)

        gt_next_pos_ratio = [r[0,0], r[1,0]]
        
        if max(gt_next_pos_ratio)>1.1:
            print(teacher_a[i][0])

        max_of_gt_next_pos_ratio= max(abs(gt_next_pos_ratio[0]), abs(gt_next_pos_ratio[1]), 1) # in [-1,1]
        gt_next_pos_ratio[0] /= max_of_gt_next_pos_ratio
        gt_next_pos_ratio[1] /= max_of_gt_next_pos_ratio

        
        teacher_a[i][0] = np.array(gt_next_pos_ratio, dtype=np.float32)
        
    return teacher_a, progress

def point_in_conrner(point, corner):
    return cv2.pointPolygonTest(corner.astype(np.float32), [point[0], point[1]], False) >= 0

def check_gt_path(path):
    error_num = 0
    for i, corner in enumerate(path):
        if i != len(path) - 1:
            next_corner = path[i+1]
            next_pos = np.mean(next_corner, axis=0)
            if not point_in_conrner(next_pos*1e6, corner*1e6):
                error_num += 1
    return error_num

def get_action(current_corner, target_corner, final_corner):
    current_pos = np.mean(current_corner, axis=0)
    target_pos = np.mean(target_corner, axis=0)
    _net_next_pos = 1e5 * (target_pos - current_pos)

    _net_y = np.round(1e5 * ((current_corner[0] + current_corner[1]) / 2 - current_pos)).astype(int)
    _net_x = np.round(1e5 * ((current_corner[1] + current_corner[2]) / 2 - current_pos)).astype(int)

    A = np.mat([
        [_net_x[0], _net_y[0]],
        [_net_x[1], _net_y[1]]
    ])
    b = np.mat([_net_next_pos[0], _net_next_pos[1]]).T
    r = np.linalg.solve(A, b)

    gt_next_pos_ratio = [r[0,0], r[1,0]]

    max_of_gt_next_pos_ratio= max(abs(gt_next_pos_ratio[0]), abs(gt_next_pos_ratio[1]), 1) # in [-1,1]
    gt_next_pos_ratio[0] /= max_of_gt_next_pos_ratio
    gt_next_pos_ratio[1] /= max_of_gt_next_pos_ratio
    offset = np.array(gt_next_pos_ratio, dtype=np.float32)

    return dict(
        altitude=float((np.linalg.norm(target_corner[0] - target_corner[1]) * 11.13 * 1e4 -40 )/ (400 - 40)),
        offset=offset,
        direction=(math.atan2(offset[0], offset[1]) / 3.14159 + 2) / 2 % 1,
        distance=np.linalg.norm(offset) * (np.linalg.norm(current_corner[0] - current_corner[1]) / 2),
        progress=compute_iou(current_corner, final_corner)
    )

def get_corner_altitude(corner):
    return float((np.linalg.norm(corner[0] - corner[1]) * 11.13 * 1e4 -40 )/ (400 - 40))

def get_origin_action(current_corner, gt_path_corners, final_corner, grid_size=27, altitude_grid_size=10):
    final_pos = final_corner.mean(axis=0)
    current_pos = current_corner.mean(axis=0)

    min_dis = 1000
    for j in range(len(gt_path_corners)-1, -1, -1):
        gt_pos = np.mean(gt_path_corners[j], axis=0)
        dis_to_current = np.linalg.norm(gt_pos - current_pos)
        if dis_to_current + 0.00001 < min_dis: # 0.00001 is in case there are two gt_path_corner are the same
            min_dis = dis_to_current
            closest_step_index = j
    altitude = get_corner_altitude(gt_path_corners[closest_step_index])
    
    if altitude_grid_size is not None:
        grid_altitude = int((altitude - 0.5/ altitude_grid_size) * altitude_grid_size)
        altitude = grid_altitude * (1/altitude_grid_size) + (0.5/altitude_grid_size)
    else:
        grid_altitude = None
    shapely_poly = shapely.geometry.Polygon(current_corner)

    line = [np.mean(corner, axis = 0) for corner in gt_path_corners]
    shapely_line = shapely.geometry.LineString(line)
    if type(shapely_poly.intersection(shapely_line)) == shapely.geometry.linestring.LineString:
        intersection_line = list(shapely_poly.intersection(shapely_line).coords)
    else:
        intersection_line = []
        for line_string in shapely_poly.intersection(shapely_line):
            intersection_line += list(line_string.coords)
    
    if intersection_line == []:
        # print(f"out of line: closest idx {closest_step_index}")
        target_point_index = -1
        line = [current_pos]+[np.mean(gt_path_corners[target_point_index], axis = 0)]
        shapely_line = shapely.geometry.LineString(line)
        intersection_line = list(shapely_poly.intersection(shapely_line).coords)
    
    min_distance = 1
    for x in intersection_line:
        x = np.array(x)
        _distance = np.linalg.norm(x - final_pos) 
        if _distance < min_distance:
            min_distance = _distance
            target_pos = x

    _net_next_pos = 1e5 * (target_pos - current_pos)
    _net_y = np.round(1e5 * ((current_corner[0] + current_corner[1]) / 2 - current_pos)).astype(int)
    _net_x = np.round(1e5 * ((current_corner[1] + current_corner[2]) / 2 - current_pos)).astype(int)

    A = np.mat([
        [_net_x[0], _net_y[0]],
        [_net_x[1], _net_y[1]]
    ])
    b = np.mat([_net_next_pos[0], _net_next_pos[1]]).T
    r = np.linalg.solve(A, b)

    gt_next_pos_ratio = [r[0,0], r[1,0]]
    
    if max(gt_next_pos_ratio)>1.1:
        print(target_pos)

    max_of_gt_next_pos_ratio= max(abs(gt_next_pos_ratio[0]), abs(gt_next_pos_ratio[1]), 1) # in [-1,1]
    gt_next_pos_ratio[0] /= max_of_gt_next_pos_ratio
    gt_next_pos_ratio[1] /= max_of_gt_next_pos_ratio
    offset = np.array(gt_next_pos_ratio, dtype=np.float32)
    
    if grid_size is not None:
        grid, offset = coordinate_to_grid(offset[0], offset[1], grid_size=grid_size)
    else:
        grid = None
    distance_progress = np.linalg.norm(current_pos - final_pos) / np.linalg.norm(gt_path_corners[0].mean(axis=0) - final_pos)

    return dict(
        altitude=altitude,
        grid_altitude=grid_altitude,
        offset=offset.tolist(),
        grid=grid,
        direction=(math.atan2(offset[0], offset[1]) / 3.14159 + 2) / 2 % 1,
        distance=np.linalg.norm(offset) * (np.linalg.norm(current_corner[0] - current_corner[1]) / 2),
        progress=compute_iou(current_corner, final_corner),
        distance_progress=distance_progress,
    )   
    
def coordinate_to_grid(x, y, grid_size=27):
    """
    将 [-1, 1] 范围内的 (x, y) 坐标划分到指定数量的格子中
    
    参数:
    x: float - 输入的 x 坐标，范围在 [-1, 1] 之间
    y: float - 输入的 y 坐标，范围在 [-1, 1] 之间
    grid_size: int - 格子的尺寸数（默认值为27）
    
    返回:
    tuple - 格子中的 (grid_x, grid_y) 坐标
    """
    # 将 [-1, 1] 转换到 [0, 1]
    x_new = (x + 1) / 2
    y_new = (y + 1) / 2
    
    # 计算格子坐标
    grid_x = int(x_new * grid_size)
    grid_y = int(y_new * grid_size)
    
    # 处理边界情况
    if grid_x == grid_size:
        grid_x = grid_size - 1
    if grid_y == grid_size:
        grid_y = grid_size - 1
    
    # 计算对应格子的中心坐标
    grid_width = 2 / grid_size
    center_x = (grid_x + 0.5) * grid_width - 1
    center_y = (grid_y + 0.5) * grid_width - 1
    
    return (grid_x, grid_y), np.array([center_x, center_y])

    
def crop_image_from_corners(img, path_corners, directions, source, resize_shape=(504, 504)):
    image_corners = []
    for corner in path_corners:
        corner = np.array(corner)
        image_corners.append(corner_gps_to_img(corner, source)) 
    mask = create_corners_mask(img.shape, image_corners)
    mask_image = apply_mask(img, mask)
    crop_image, bounding_box = crop_to_bounding_box(mask_image, mask)
    
    if resize_shape is not None:
        crop_image = cv2.resize(crop_image, resize_shape, interpolation=cv2.INTER_AREA)
    
    image_corners_new = []
    for corner in image_corners:
        image_corners_new.append(convert_coordinates(corner, bounding_box, img.shape[:2], crop_image.shape[:2]))
    corner_info = []
    for t, corner in enumerate(image_corners_new):
        corner.sort()
        corner = np.round(np.array(corner)*(100/(crop_image.shape[0]-1))).astype(int).tolist()
        corner_info.append((corner[0], corner[-1], directions[t]))
    
    return crop_image, corner_info

def get_corner_prompt(corner_info):
    prompt = ""
    length = len(corner_info)
    for i in range(length):
        left_top, right_bottom, direction = corner_info[i]
        prompt += f"observation {i}: {{<{left_top[0]}><{left_top[1]}><{right_bottom[0]}><{right_bottom[1]}>|<{direction}>}}"
        if i != length - 1:
            prompt += ","
        else:
            prompt += ""
    prompt = f"[{prompt}]"
    return prompt

def postprocess_pred_res(pred_next_pos_ratio, pred_altitude, pred_progress, pred_distance_progress=None):
    # Predicted progress
    pred_progress_t = pred_progress.cpu().detach().float().numpy() 

    # Predicted waypoint
    a_t_next_pos_ratio = pred_next_pos_ratio.cpu().detach().float().numpy()
    for i in range(len(a_t_next_pos_ratio)):
        max_of_a_t_next_pos_i = max(abs(a_t_next_pos_ratio[i][0]), abs(a_t_next_pos_ratio[i][1]), 1)
        a_t_next_pos_ratio[i][0] /= max_of_a_t_next_pos_i
        a_t_next_pos_ratio[i][1] /= max_of_a_t_next_pos_i

    # Predicted altitude
    a_t_altitude = pred_altitude.cpu().detach().float().numpy() 

    # Clip the prediction to (0,1)
    for i in range(len(a_t_altitude)):
        a_t_altitude[i] = min(1., max(0., a_t_altitude[i]))
    for i in range(len(pred_progress_t)):
        pred_progress_t[i] = min(1., max(0., pred_progress_t[i]))

    if pred_distance_progress is not None:
        pred_distance_progress = pred_distance_progress.cpu().detach().float().numpy()
        for i in range(len(pred_distance_progress)):
            pred_distance_progress[i] = min(1., max(0., pred_distance_progress[i]))

    target_list = []
    for i in range(len(a_t_altitude)):
        offset = a_t_next_pos_ratio[i]
        altitude = a_t_altitude[i]
        progress = pred_progress_t[i]
        distance_progress = pred_distance_progress[i]
        target = dict(
            offset=offset,
            altitude=altitude[0],
            progress=progress[0],
            distance_progress=distance_progress[0],
        )

        target_list.append(target)
        # direction = (math.atan2(offset[0], offset[1]) /3.14159 + 2) / 2 % 1
        # distance = np.linalg.norm(offset) * (np.linalg.norm(current_corner[0] - current_corner[1])/2)
    return target_list

def eval_item(gt_path, gt_corners, path, corners, progress):
    scores = {}
    scores['trajectory_lengths'] = np.sum([np.linalg.norm(a-b) for a, b in zip(path[:-1], path[1:])])
    scores['trajectory_lengths'] = scores['trajectory_lengths']*11.13*1e4
    gt_whole_lengths =  np.sum([np.linalg.norm(a-b) for a, b in zip(gt_path[:-1], gt_path[1:])])*11.13*1e4
    gt_net_lengths =  np.linalg.norm(gt_path[0] - gt_path[-1]) *11.13*1e4


    scores['iou'] = progress[-1] # same as compute_iou(corners[-1], gt_corners[-1]）

    scores['gp'] = gt_net_lengths - \
                    np.linalg.norm(path[-1] - gt_path[-1])*11.13*1e4
    scores['oracle_gp'] = gt_net_lengths - \
                    np.min([np.linalg.norm(path[x] - gt_path[-1]) for x in range(len(path)) ])*11.13*1e4

    scores['success'] = float(progress[-1] >= 0.4)
    _center = np.mean(gt_corners[-1], axis=0) 
    _point = Point(_center)
    _poly = Polygon(np.array(corners[-1]))
    if not _poly.contains(_point):
        scores['success'] = float(0)
    
    _center = np.mean(corners[-1], axis=0) 
    _point = Point(_center)
    _poly = Polygon(np.array(gt_corners[-1]))
    if not _poly.contains(_point):
        scores['success'] = float(0)


    scores['oracle_success'] = float(any(np.array(progress) > 0.4))
    scores['gt_length'] = gt_whole_lengths
    scores['spl'] = scores['success'] * gt_net_lengths / max(scores['trajectory_lengths'], gt_net_lengths, 0.01)
    return scores

def eval_metrics(preds, human_att_eval=False):
    ''' 
    Evaluate each agent trajectory based on how close it got to the goal location 
    the path contains [view_id, angle, vofv]
    '''
    # print('eval %d predictions' % (len(preds)))

    metrics = defaultdict(list)
    if human_att_eval == True:
        for k in preds.keys():
            if 'human_att_performance' in preds[k].keys():
                metrics['human_att_performance']+=preds[k]['human_att_performance']
                nss = np.mean(preds[k]['nss'])
                if nss == nss:
                    metrics['nss'].append(nss)
        metrics['human_att_performance'] = np.mean(metrics['human_att_performance'], axis=0)
        metrics['nss'] = np.mean(metrics['nss'])
        if metrics['nss'] == metrics['nss']: 
            avg_metrics = {"HA_precision": metrics['human_att_performance'][0],
                            "HA_recall": metrics['human_att_performance'][0],
                            "nss": metrics['nss']}
        else: 
            avg_metrics = {"HA_precision": 0,
                            "HA_recall": 0,
                            "nss":0}
        return avg_metrics, metrics

    for k in preds.keys():
        item = preds[k]
        instr_id = item['instr_id']
        # print(instr_id)
        dia_number = 0
        if 'num_dia' in preds[k].keys():
            dia_number = preds[k]['num_dia']
        traj = [np.mean(x, axis = 0) for x in item['path_corners']]      # x = (corners, directions)
        corners = [np.array(x) for x in item['path_corners']]      # x = (corners, directions)
        progress = [x for x in item['gt_progress']]
        gt_corners = [np.array(x) for x in item['gt_path_corners']]
        gt_trajs = [np.mean(x, axis = 0) for x in item['gt_path_corners']]
        
        traj_scores = eval_item(gt_trajs, gt_corners, traj, corners, progress)
        for k, v in traj_scores.items():
            if k == 'iou' and traj_scores['success']:
                metrics[k].append(v)
            else:
                metrics[k].append(v)

        if dia_number == 1:
            metrics['success_1'].append(traj_scores['success'])  
            metrics['spl_1'].append(traj_scores['spl'])
            metrics['gp_1'].append(traj_scores['gp'])
        elif dia_number == 2:
            metrics['success_2'].append(traj_scores['success'])  
            metrics['spl_2'].append(traj_scores['spl'])
            metrics['gp_2'].append(traj_scores['gp'])
        else:
            metrics['success_else'].append(traj_scores['success'])  
            metrics['spl_else'].append(traj_scores['spl'])
            metrics['gp_else'].append(traj_scores['gp'])
            
        if traj_scores['trajectory_lengths'] > 150:
            metrics['success_long'].append(traj_scores['success'])  
            metrics['spl_long'].append(traj_scores['spl'])
            metrics['gp_long'].append(traj_scores['gp'])
        else:
            metrics['success_short'].append(traj_scores['success'])  
            metrics['spl_short'].append(traj_scores['spl'])
            metrics['gp_short'].append(traj_scores['gp'])
        metrics['instr_id'].append(instr_id)

    avg_metrics = {
        # 'steps': np.mean(metrics['trajectory_steps']),
        'lengths': np.mean(metrics['trajectory_lengths']),
        'sr': np.mean(metrics['success']) * 100,
        'oracle_sr': np.mean(metrics['oracle_success']) * 100,
        'spl': np.mean(metrics['spl']) * 100,
        'gp': np.mean(metrics['gp']),
        'oracle_gp': np.mean(metrics['oracle_gp']),
        'gt_length': np.mean(metrics['gt_length']),
        'iou' : np.mean(metrics['iou']),
        # 'spl_short': np.mean(metrics['spl_short']) * 100,
        # 'sr_short': np.mean(metrics['success_short']) * 100,
        # 'gp_short': np.mean(metrics['gp_short']),
    }
    if len(metrics['success_1']) != 0:
        avg_metrics['num_1']= len(metrics['success_1'])
        avg_metrics['spl_1']= np.mean(metrics['spl_1']) * 100
        avg_metrics['sr_1']=np.mean(metrics['success_1']) * 100
        avg_metrics['gp_1']=np.mean(metrics['gp_1'])

    if len(metrics['success_2']) != 0:
        avg_metrics['num_2']= len(metrics['success_2'])
        avg_metrics['spl_2']= np.mean(metrics['spl_2']) * 100
        avg_metrics['sr_2']=np.mean(metrics['success_2']) * 100
        avg_metrics['gp_2']=np.mean(metrics['gp_2'])

    if len(metrics['success_else']) != 0:
        avg_metrics['num_else']= len(metrics['success_else'])
        avg_metrics['spl_else']= np.mean(metrics['spl_else']) * 100
        avg_metrics['sr_else']=np.mean(metrics['success_else']) * 100
        avg_metrics['gp_else']=np.mean(metrics['gp_else'])
    
    return avg_metrics, metrics

def transforms_augment_data(aug_data_path, origin_data_path):
    pass


def get_map_info_dict(json_data):
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
    return map_dict

def get_dialog_data(map_dict):
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
    return full_dialog_data