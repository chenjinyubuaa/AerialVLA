import argparse
import torch
import os
import json
from tqdm import tqdm
import numpy as np

from typing import Dict, Optional, Sequence, List
import transformers
import re
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def show_box(box, ax, edgecolor='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2))  

def show_boxs(image, box_coords=None, caption=None, save_path="test.png"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, box in enumerate(box_coords):
        if i % 2 == 0:
            show_box(box, plt.gca(), edgecolor='green')
        else:
            show_box(box, plt.gca(), edgecolor='red')

    if caption is not None:
        plt.title(caption, fontsize=18)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(hspace=0)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def evaluate(args):
    answer_file = args.answers_file
    res = args.res
    image_folder = args.image_folder

    with open(answer_file, "r") as file:
         for line in file:
            line = line.strip()
            answer = json.loads(line)

            pred_boxes = answer['answer']
            gt_boxes = answer['ground_truth']
            image_id = answer['image_id']
            question = answer['question']

            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\|<\d{1,3}>\}'
            matches = re.findall(pattern, pred_boxes)

            test_boxes = []
            for i, match in enumerate(matches):
                integers = re.findall(r'\d+', match)
                pred_bbox = [int(num) for num in integers]
                image_path = os.path.join(image_folder, image_id+'.png')
                image = Image.open(image_path)
                width, height = image.size

                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                try:
                    gt_box = gt_boxes[i]
                except:
                    break
                test_boxes.extend([
                    [*gt_box[0], *gt_box[2]],
                    pred_bbox[0:4]
                ])
            show_boxs(image, test_boxes, question)
            breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/lihongyu/projects/aerial_chat/datasets/huggingface.co/datasets/MBZUAI/GeoChat_Instruct/share/softwares/kartik/GeoChat_finetuning/final_images_llava")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--res", type=float, default=100.)
    args = parser.parse_args()

    evaluate(args)