# 根目录
ROOT = '/home/wong/Documents/DataSet/CityPerson/'
# COCO format标注save位置
COCO_ANNO_DIR = ROOT + 'annotations/'

TARGET_TRAIN_DIR = ROOT + 'train/'
TARGET_VAL_DIR = ROOT + 'val/'
TARGET_TEST_DIR = ROOT + 'test/'

import json
import os

def check(json_file_path, folder_path):
    with open(json_file_path, 'r') as anno_file:
        anno_data = json.load(anno_file)
    image_list = anno_data['images']
    count = 0
    for im in image_list:
        im_path = folder_path + im['file_name']
        if not os.path.isfile(im_path):
            print("ERROR: NOT FOUND " + im_path)
            return False
        else:
            count += 1
    print(f"CHECKED {count} images in {folder_path} with {json_file_path} \n")
    return True

check(COCO_ANNO_DIR + 'custom_train_debug.json', TARGET_TRAIN_DIR)
check(COCO_ANNO_DIR + 'custom_val_debug.json', TARGET_VAL_DIR)

check(COCO_ANNO_DIR + 'custom_train.json', TARGET_TRAIN_DIR)
check(COCO_ANNO_DIR + 'custom_val.json', TARGET_VAL_DIR)
check(COCO_ANNO_DIR + 'custom_test.json', TARGET_TEST_DIR)