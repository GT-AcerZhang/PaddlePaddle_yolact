#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-01-23 15:16:15
#   Description : paddlepaddle_yolact++
#                 对coco2017 test-dev进行预测，生成json文件。
#
# ================================================================
import cv2
import os
import time
import json
import shutil
import sys
import numpy as np
import pycocotools

from model.decode import Decode


coco_cats = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
coco_cats_inv = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]
class Detections:
    def __init__(self):
        self.bbox_data = []
        self.mask_data = []
    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        # 如果出现module 'pycocotools' has no attribute 'mask'
        # 找到pycocotools安装路径下的__init__.py， 加上from .mask import *
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })

    def dump(self, image_id):
        dump_arguments = [
            (self.bbox_data, 'results/bbox/%.12d.json' % image_id),
            (self.mask_data, 'results/mask/%.12d.json' % image_id)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)

    def dump_web(self, bbox_data, mask_data):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        output = {
            'info': {
                'Config': {
                    'preserve_aspect_ratio': False,
                    'use_prediction_module': False,
                    'use_yolo_regressors': False,
                    'use_prediction_matching': False,  # 是否使用预测框决定正反例。不是，用的是先验框。
                    'train_masks': True
                }
            }
        }

        image_ids = list(set([x['image_id'] for x in bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(bbox_data, mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': COCO_CLASSES[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join('results/', 'yolact_plus_resnet50.json'), 'w') as f:
            json.dump(output, f)


if __name__ == '__main__':
    file = 'data/coco_classes.txt'

    # model_path = 'pretrained_resnet50'
    # model_path = 'pretrained_resnet101'
    # model_path = 'pretrained_resnet50dcn'
    # model_path = 'pretrained_darknet53'
    model_path = './weights/step1080600-ep074-loss4.179.pd'

    backbone_names = ['resnet50', 'resnet101', 'resnet50dcn', 'darknet53']
    backbone_name = backbone_names[2
    ]

    use_gpu = True
    import platform
    sysstr = platform.system()
    if sysstr == 'Windows':
        use_gpu = False
        # use_gpu = True

    _decode = Decode(backbone_name, 550, 0.05, 0.5, model_path, file, use_gpu=use_gpu)

    if os.path.exists('results/bbox/'): shutil.rmtree('results/bbox/')
    if os.path.exists('results/mask/'): shutil.rmtree('results/mask/')
    if not os.path.exists('results/'): os.mkdir('results/')
    os.mkdir('results/bbox/')
    os.mkdir('results/mask/')

    # 跑test-dev
    images = None
    with open('../data/data7122/annotations/image_info_test-dev2017.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            new_dict = json.loads(line)
            images = new_dict['images']

    root = '../data/data7122/test2017/'
    # root = '../COCO/test2017/'
    p = 0
    num_imgs = len(images)
    start = time.time()
    for img in images:
        image_id = img['id']
        file_name = img['file_name']

        path = os.path.join(root, file_name)
        image = cv2.imread(path)
        p += 1
        # 自定义进度条
        percent = (p / num_imgs) * 100
        num = int(29 * percent / 100)
        ETA = int((time.time() - start) * (100 - percent) / percent)
        sys.stdout.write('\r{0}'.format(' ' * (len(str(num_imgs)) - len(str(p)))) + \
            '{0}/{1} [{2}>'.format(p, num_imgs, '=' * num) + '{0}'.format(
            '.' * (29 - num)) + ']' + ' - ETA: ' + str(ETA) + 's')
        sys.stdout.flush()


        image, boxes, masks, classes, scores = _decode.detect_image(image, draw=False)
        if len(boxes) == 0:
            continue
        detections = Detections()
        for i in range(masks.shape[0]):
            # Make sure that the bounding box actually makes sense and a mask was produced
            if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                detections.add_bbox(image_id, classes[i], boxes[i, :], scores[i])
                detections.add_mask(image_id, classes[i], masks[i, :, :], scores[i])   # 使用预测框得分
        detections.dump(image_id)
    print('\ntotal time: {0:.6f}s'.format(time.time() - start))

    print('Generating json file...')
    bbox_list = []
    path_dir = os.listdir('results/bbox/')
    for name in path_dir:
        with open('results/bbox/'+name, 'r', encoding='utf-8') as f2:
            for line in f2:
                line = line.strip()
                r_list = json.loads(line)
                bbox_list += r_list
    with open('results/bbox_detections.json', 'w') as f2:
        json.dump(bbox_list, f2)
    mask_list = []
    path_dir = os.listdir('results/mask/')
    for name in path_dir:
        with open('results/mask/'+name, 'r', encoding='utf-8') as f2:
            for line in f2:
                line = line.strip()
                r_list = json.loads(line)
                mask_list += r_list
    with open('results/mask_detections.json', 'w') as f2:
        json.dump(mask_list, f2)
    # 提交到网站的文件
    with open('results/detections_test-dev2017_yolactplus_results.json', 'w') as f2:
        json.dump(mask_list, f2)



