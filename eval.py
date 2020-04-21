#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-01-23 15:16:15
#   Description : paddlepaddle_yolact++
#                 对val2017计算mAP
#
# ================================================================
import time
import sys
import numpy as np
from collections import OrderedDict

from data.coco import COCODetection
from train import jaccard_np
from utils import BaseTransform, get_transform, MEANS, STD
from model.decode import Decode


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

# 10种iou阈值
iou_thresholds = [x / 100 for x in range(50, 100, 5)]

def mask_iou(masks_a, masks_b, iscrowd=False):
    """
    Computes the pariwise mask IoU between two sets of masks of shape [a, h, w] and [b, h, w].
    The output is of shape [a, b].
    """
    masks_a = np.reshape(masks_a, (np.shape(masks_a)[0], -1))   # (a, h*w)
    masks_b = np.reshape(masks_b, (np.shape(masks_b)[0], -1))   # (b, h*w)

    intersection = np.matmul(masks_a, masks_b.transpose(1, 0))   # (a, b)
    area_a = np.sum(masks_a, axis=1, keepdims=True)   # (a, 1)
    area_b = np.sum(masks_b, axis=1, keepdims=True)   # (b, )
    area_b = np.reshape(area_b, (1, -1))              # (1, b)

    return intersection / (area_a + area_b - intersection) if not iscrowd else intersection / area_a



def _mask_iou(mask1, mask2, iscrowd=False):
    ret = mask_iou(mask1, mask2, iscrowd)
    return ret

def _bbox_iou(bbox1, bbox2, iscrowd=False):
    ret = jaccard_np(bbox1, bbox2, iscrowd)
    return ret

def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(COCO_CLASSES)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values() ) -1))

    print_maps(all_maps)

    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)



if __name__ == '__main__':
    file = 'data/coco_classes.txt'

    # model_path = 'pretrained_resnet50'
    # model_path = 'pretrained_resnet101'
    # model_path = 'pretrained_resnet50dcn'
    # model_path = 'pretrained_darknet53'
    model_path = './weights/step829200-ep057-loss6.409.pd'

    backbone_names = ['resnet50', 'resnet101', 'resnet50dcn', 'darknet53']
    backbone_name = backbone_names[2
    ]

    use_gpu = True
    import platform
    sysstr = platform.system()
    if sysstr == 'Windows':
        use_gpu = False
        # use_gpu = True

    _decode = Decode(backbone_name, 550, 0.05, 0.5, model_path, file, use_gpu=use_gpu, use_fast_prep=False)

    ap_data = {
        'box': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds],
        'mask': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thresholds]
    }

    valid_images = '../COCO/val2017/'
    valid_info = '../COCO/annotations/instances_val2017.json'
    transform = get_transform(backbone_name)
    val_dataset = COCODetection(image_path=valid_images,
                                info_file=valid_info,
                                transform=BaseTransform(transform, 550, MEANS, STD))  # 不使用数据增强

    num_imgs = len(val_dataset)
    start = time.time()
    # 不管训练集还是验证集，坐标gt(x0, y0, x1, y1)都是归一化后的值。
    # 训练集的gt_masks resize成550x550，而验证集的gt_masks没有做resize。
    for image_idx in range(num_imgs):
        img, gt, gt_masks, num_crowd = val_dataset.pull_item(image_idx)
        _, h, w = gt_masks.shape   # 因为验证集的gt_masks没有做resize。所以从这里获得原图高宽
        boxes, masks, classes, scores = _decode.eval_image(img, h, w)
        if len(boxes) == 0:
            continue
        box_scores = scores
        mask_scores = scores
        masks = np.reshape(masks, (-1, h*w))


        # 作者写好的代码，按照格式提供变量即可
        gt_boxes = gt[:, :4]
        gt_boxes[:, [0, 2]] *= w
        gt_boxes[:, [1, 3]] *= h
        gt_classes = list(gt[:, 4].astype(int))
        gt_masks = np.reshape(gt_masks, (-1, h*w))

        if num_crowd > 0:
            split = lambda x: (x[-num_crowd:], x[:-num_crowd])
            crowd_boxes, gt_boxes = split(gt_boxes)
            crowd_masks, gt_masks = split(gt_masks)
            crowd_classes, gt_classes = split(gt_classes)
        num_pred = len(classes)
        num_gt = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes, gt_boxes)

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes, crowd_boxes, iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])  # box得分递减排序的得分下标
        # mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])
        mask_indices = sorted(range(num_pred), key=lambda i: -mask_scores[i])  # mask得分递减排序的得分下标

        iou_types = [
            ('box', lambda i, j: bbox_iou_cache[i, j].item(),
             lambda i, j: crowd_bbox_iou_cache[i, j].item(),
             lambda i: box_scores[i], box_indices),
            ('mask', lambda i, j: mask_iou_cache[i, j].item(),
             lambda i, j: crowd_mask_iou_cache[i, j].item(),
             lambda i: mask_scores[i], mask_indices)
        ]

        # for _class in set(classes + gt_classes):  # [0, 1]   [1, 2]
        for _class in set([*classes, *gt_classes]):  # [0, 1]   [1, 2]
            ap_per_iou = []
            num_gt_for_class = sum([1 for x in gt_classes if x == _class])  # gt里有这一类别的样本的数量

            for iouIdx in range(len(iou_thresholds)):
                iou_threshold = iou_thresholds[iouIdx]

                for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                    gt_used = [False] * len(gt_classes)

                    ap_obj = ap_data[iou_type][iouIdx][_class]
                    ap_obj.add_gt_positives(num_gt_for_class)

                    for i in indices:
                        if classes[i] != _class:
                            continue

                        max_iou_found = iou_threshold
                        max_match_idx = -1
                        for j in range(num_gt):
                            if gt_used[j] or gt_classes[j] != _class:
                                continue

                            iou = iou_func(i, j)

                            if iou > max_iou_found:
                                max_iou_found = iou
                                max_match_idx = j

                        if max_match_idx >= 0:
                            gt_used[max_match_idx] = True
                            ap_obj.push(score_func(i), True)
                        else:
                            # If the detection matches a crowd, we can just ignore it
                            matched_crowd = False

                            if num_crowd > 0:
                                for j in range(len(crowd_classes)):
                                    if crowd_classes[j] != _class:
                                        continue

                                    iou = crowd_func(i, j)

                                    if iou > iou_threshold:
                                        matched_crowd = True
                                        break

                            # All this crowd code so that we can make sure that our eval code gives the
                            # same result as COCOEval. There aren't even that many crowd annotations to
                            # begin with, but accuracy is of the utmost importance.
                            if not matched_crowd:
                                ap_obj.push(score_func(i), False)
        # 自定义进度条
        percent = ((image_idx + 1) / num_imgs) * 100
        num = int(29 * percent / 100)
        ETA = int((time.time() - start) * (100 - percent) / percent)
        sys.stdout.write('\r{0}'.format(' ' * (len(str(num_imgs)) - len(str((image_idx + 1))))) + \
            '{0}/{1} [{2}>'.format((image_idx + 1), num_imgs, '=' * num) + '{0}'.format(
            '.' * (29 - num)) + ']' + ' - ETA: ' + str(ETA) + 's')
        sys.stdout.flush()
    print('\ntotal time: {0:.6f}s'.format(time.time() - start))
    calc_map(ap_data)


