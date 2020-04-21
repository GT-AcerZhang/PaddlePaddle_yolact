#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-01-23 15:16:15
#   Description : paddlepaddle_yolact++
#
# ================================================================
import paddle.fluid.layers as P

import paddle.fluid as fluid
import random
import colorsys
import cv2
import time
import os
import numpy as np
from model.yolact import Yolact
from utils.tools import get_transform, get_strides, get_priors


def sanitize_coordinates_np(_x1, _x2, img_size:int):
    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.maximum(x1, np.zeros(x1.shape))
    x2 = np.minimum(x2, np.ones(x2.shape))
    x1 = x1 * img_size
    x2 = x2 * img_size
    return x1, x2

def point_form(boxes):
    # (cx, cy, w, h)变成(x0, y0, x1, y1)
    return P.concat([boxes[:, :2] - boxes[:, 2:] * 0.5,
                     boxes[:, :2] + boxes[:, 2:] * 0.5], axis=-1)

def decode(pred_txtytwth, priors, use_yolo_regressors: bool = False):
    """ 对神经网络预测的坐标tx、ty、tw、th进行解码。默认用的是SSD的解码方式 """
    if use_yolo_regressors:
        # Decoded boxes in center-size notation
        boxes = P.concat([
            pred_txtytwth[:, :2] + priors[:, :2],
            priors[:, 2:] * P.exp(pred_txtytwth[:, 2:])
        ], 1)

        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]

        boxes = P.concat([
            priors[:, :2] + pred_txtytwth[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * P.exp(pred_txtytwth[:, 2:] * variances[1])], 1)
        x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
        x2y2 = boxes[:, :2] + boxes[:, 2:] / 2

    return P.concat([x1y1, x2y2], 1)


def intersect(box_a, box_b):   # 相交区域的面积
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = P.shape(box_a)[0]
    A = P.shape(box_a)[1]
    B = P.shape(box_b)[1]

    box_a = P.reshape(box_a, (n, A, 1, 4))
    box_b = P.reshape(box_b, (n, 1, B, 4))
    expand_box_a = P.expand(box_a, [1, 1, B, 1])
    expand_box_b = P.expand(box_b, [1, A, 1, 1])

    # 相交矩形的左上角坐标、右下角坐标
    left_up = P.elementwise_max(expand_box_a[:, :, :, :2], expand_box_b[:, :, :, :2])
    right_down = P.elementwise_min(expand_box_a[:, :, :, 2:], expand_box_b[:, :, :, 2:])

    inter_section = P.relu(right_down - left_up)
    return inter_section[:, :, :, 0] * inter_section[:, :, :, 1]

def jaccard(box_a, box_b, iscrowd: bool = False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if len(box_a.shape) == 2:
        use_batch = False
        box_a = P.reshape(box_a, (1, P.shape(box_a)[0], P.shape(box_a)[1]))
        box_b = P.reshape(box_b, (1, P.shape(box_b)[0], P.shape(box_b)[1]))

    inter = intersect(box_a, box_b)

    area_a = (box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])
    area_a = P.reshape(area_a, (P.shape(area_a)[0], P.shape(area_a)[1], 1))
    area_a = P.expand(area_a, [1, 1, P.shape(inter)[2]])

    area_b = (box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])
    area_b = P.reshape(area_b, (P.shape(area_b)[0], 1, P.shape(area_b)[1]))
    area_b = P.expand(area_b, [1, P.shape(inter)[1], 1])

    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out[0]


def sanitize_coordinates(_x1, _x2, img_size, padding: int = 0, cast: bool = True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    x1 = P.elementwise_min(_x1, _x2)
    x2 = P.elementwise_max(_x1, _x2)
    x1 = P.relu(x1 - padding)        # 下限是0
    img_size2 = P.expand(img_size, (P.shape(x2)[0], ))
    img_size2 = P.cast(img_size2, 'float32')
    x2 = img_size2 - P.relu(img_size2 - (x2 + padding))   # 上限是img_size
    if cast:
        x1 = P.cast(x1, 'int32')
        x2 = P.cast(x2, 'int32')
    return x1, x2

def crop(masks, boxes, padding: int = 1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks    。n是正样本数量
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = P.shape(masks)[0], P.shape(masks)[1], P.shape(masks)[2]
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = P.range(0, w, 1, 'int32')
    cols = P.range(0, h, 1, 'int32')
    rows = P.expand(P.reshape(rows, (1, -1, 1)), [h, 1, n])
    cols = P.expand(P.reshape(cols, (-1, 1, 1)), [1, w, n])
    rows.stop_gradient = True
    cols.stop_gradient = True

    x1 = P.reshape(x1, (1, 1, -1))
    x2 = P.reshape(x2, (1, 1, -1))
    y1 = P.reshape(y1, (1, 1, -1))
    y2 = P.reshape(y2, (1, 1, -1))
    x1.stop_gradient = True
    x2.stop_gradient = True
    y1.stop_gradient = True
    y2.stop_gradient = True
    masks_left = P.cast(rows >= P.expand(x1, [h, w, 1]), 'float32')
    masks_right = P.cast(rows < P.expand(x2, [h, w, 1]), 'float32')
    masks_up = P.cast(cols >= P.expand(y1, [h, w, 1]), 'float32')
    masks_down = P.cast(cols < P.expand(y2, [h, w, 1]), 'float32')

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, conf_thresh=0.05, nms_thresh=0.5, top_k=200):
        self.background_label = 0
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh

    def __call__(self, predictions):
        loc_data = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        prior_data = predictions['priors']

        conf_preds = P.transpose(conf_data, perm=[0, 2, 1])
        decoded_boxes = decode(loc_data[0], prior_data)
        boxes, masks, classes, scores = self.detect(0, conf_preds, decoded_boxes, mask_data)

        masks = P.matmul(predictions['proto'][0], masks, transpose_y=True)
        masks = P.sigmoid(masks)
        masks = crop(masks, boxes)

        return boxes, masks, classes, scores

    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        # 确实是先坐标全部解码完成，在进行分数过滤。可以考虑过滤后再进行坐标解码
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores = P.reduce_max(cur_scores, dim=0)

        '''
        gpu版本的paddlepaddle1.6.2里有一个问题。keep如果是[None]，并且在gather()里使用了keep，就会出现
        cudaGetLastError  invalid configuration argument errno: 9   这个错误。cpu版本则可以正常跑。
        为了避免上面的问题，只能让keep不是[None]，所以这里给keep额外添加了一个元素keep_extra。
        '''
        keep = P.where(conf_scores > self.conf_thresh)
        keep_extra = P.where(conf_scores < self.conf_thresh)
        keep_extra = keep_extra[:1]
        keep = P.concat([keep, keep_extra], axis=0)
        scores = P.gather(P.transpose(cur_scores, perm=[1, 0]), keep)
        scores = P.transpose(scores, perm=[1, 0])
        boxes = P.gather(decoded_boxes, keep)
        masks = P.gather(mask_data[batch_idx], keep)

        '''
        因为上面增加了一个keep_extra，所以keep一定至少有一个预测框。
        当官方修复了上述问题后，删除上面keep_extra的代码，下面的代码解除注释。
        这么做的原因是判断keep为空太难了。
        '''
        # 可能没有框被保留。所以添加一个得分垫底的框让fast_nms()能进行下去
        # extra_box = P.fill_constant((1, 4), 'float32', value=-1.0)
        # extra_score = P.fill_constant((P.shape(cur_scores)[0], 1), 'float32', value=-1.0)
        # extra_mask = P.fill_constant((1, P.shape(mask_data)[2]), 'float32', value=-1.0)
        # boxes = P.concat([boxes, extra_box], axis=0)
        # scores = P.concat([scores, extra_score], axis=1)
        # masks = P.concat([masks, extra_mask], axis=0)

        return self.fast_nms(boxes, scores, masks)

    def fast_nms(self, boxes, scores, masks, max_num_detections=100):
        iou_threshold = self.nms_thresh
        top_k = self.top_k

        # 同类方框根据得分降序排列
        scores, idx = P.argsort(scores, axis=1, descending=True)

        idx = idx[:, :top_k]
        scores = scores[:, :top_k]

        num_classes, num_dets = P.shape(idx)[0], P.shape(idx)[1]

        idx = P.reshape(idx, (-1, ))
        boxes = P.gather(boxes, idx)
        boxes = P.reshape(boxes, (num_classes, num_dets, 4))
        masks = P.gather(masks, idx)
        masks = P.reshape(masks, (num_classes, num_dets, -1))

        # 计算一个c×n×n的IOU矩阵，其中每个n×n矩阵表示对该类n个候选框，两两之间的IOU
        iou = jaccard(boxes, boxes)
        # 因为自己与自己的IOU=1，IOU(A,B)=IOU(B,A)，所以对上一步得到的IOU矩阵
        # 进行一次处理。具体做法是将每一个通道，的对角线元素和下三角部分置为0
        rows = P.range(0, num_dets, 1, 'int32')
        cols = P.range(0, num_dets, 1, 'int32')
        rows = P.expand(P.reshape(rows, (1, -1)), [num_dets, 1])
        cols = P.expand(P.reshape(cols, (-1, 1)), [1, num_dets])
        tri_mask = P.cast(rows > cols, 'float32')
        tri_mask = P.expand(P.reshape(tri_mask, (1, num_dets, num_dets)), [num_classes, 1, 1])
        iou = tri_mask * iou
        iou_max = P.reduce_max(iou, dim=1)

        # Now just filter out the ones higher than the threshold
        keep = P.where(iou_max <= iou_threshold)

        # Assign each kept detection to its corresponding class
        classes = P.range(0, num_classes, 1, 'int32')
        classes = P.expand(P.reshape(classes, (-1, 1)), [1, num_dets])
        classes = P.gather_nd(classes, keep)

        boxes = P.gather_nd(boxes, keep)
        masks = P.gather_nd(masks, keep)
        scores = P.gather_nd(scores, keep)

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = P.argsort(scores, axis=0, descending=True)
        idx = idx[:max_num_detections]
        scores = scores[:max_num_detections]

        classes = P.gather(classes, idx)
        boxes = P.gather(boxes, idx)
        masks = P.gather(masks, idx)

        return boxes, masks, classes, scores


class Decode(object):
    def __init__(self, backbone_name, input_size, obj_threshold, nms_threshold, model_path, file_path, use_gpu, is_test=True, use_fast_prep=True):
        self._t1 = obj_threshold
        self._t2 = nms_threshold
        self.all_classes = self.get_classes(file_path)
        self.num_classes = len(self.all_classes) + 1

        mask_dim = 32   # 掩膜原型数
        if backbone_name == 'resnet50dcn':
            pred_aspect_ratios = [[[1, 1 / 2, 2]]] * 5
            pred_scales = [[i * 2 ** (j / 3.0) for j in range(3)] for i in [24, 48, 96, 192, 384]]
            use_pixel_scales = True
            preapply_sqrt = False
            use_square_anchors = False
            transform = get_transform(backbone_name)
        elif backbone_name == 'resnet50' or backbone_name == 'resnet101':
            pred_aspect_ratios = [[[1, 1 / 2, 2]]] * 5
            pred_scales = [[24], [48], [96], [192], [384]]
            use_pixel_scales = True
            preapply_sqrt = False
            use_square_anchors = True
            transform = get_transform(backbone_name)
        elif backbone_name == 'darknet53':
            pred_aspect_ratios = [[[1, 1 / 2, 2]]] * 5
            pred_scales = [[24], [48], [96], [192], [384]]
            use_pixel_scales = True
            preapply_sqrt = False
            use_square_anchors = True
            transform = get_transform(backbone_name)
        strides = get_strides(input_size)
        self.priors, class_vectors, num_priors, num_priors_list = get_priors(input_size, self.num_classes, strides,
                                                                        pred_aspect_ratios, pred_scales, preapply_sqrt,
                                                                        use_pixel_scales,
                                                                        use_square_anchors)
        if use_fast_prep:
            inputs = P.data(name='input_1', shape=[-1, -1, -1, 3], append_batch_size=False, dtype='float32')
        else:
            inputs = P.data(name='input_1', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
        priors_tensor = P.data(name='priors', shape=[-1, 4], append_batch_size=False, dtype='float32')
        pred_outs = Yolact(backbone_name, inputs, self.num_classes, mask_dim, num_priors_list, is_test, transform=transform,
                           input_size=input_size, use_fast_prep=use_fast_prep)
        pred_outs['priors'] = priors_tensor
        dt = Detect(conf_thresh=self._t1, nms_thresh=self._t2)
        self.boxes, self.masks, self.classes, self.scores = dt(pred_outs)

        # Create an executor using CPU as an example
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        self.place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)
        self.exe.run(fluid.default_startup_program())

        if model_path is not None:
            fluid.io.load_persistables(self.exe, model_path)

    # 处理一张图片
    def detect_image(self, image, top_k=100, draw=True):
        h, w, _ = image.shape
        img = np.expand_dims(image, axis=0)

        feed_dic = {'input_1': img.astype(np.float32), 'priors': self.priors.astype(np.float32)}

        boxes, masks, classes, scores = self.exe.run(fluid.default_main_program(),
                                             feed=feed_dic,
                                             fetch_list=[self.boxes, self.masks, self.classes, self.scores])

        # 把增加的多余框去掉
        keep = np.where(scores > self._t1)[0]
        boxes = boxes[keep]
        scores = scores[keep]
        masks = masks[:, :, keep]
        classes = classes[keep]
        if len(keep) != 0:
            # 保留前k个
            boxes = boxes[:top_k, :]
            scores = scores[:top_k]
            masks = masks[:, :, :top_k]
            classes = classes[:top_k]

            masks = cv2.resize(masks, (w, h), interpolation=cv2.INTER_LINEAR)
            masks = np.reshape(masks, (h, w, -1))
            masks = masks.transpose(2, 0, 1)
            masks = (masks > 0.5).astype(np.float32)
            boxes[:, 0], boxes[:, 2] = sanitize_coordinates_np(boxes[:, 0], boxes[:, 2], w)
            boxes[:, 1], boxes[:, 3] = sanitize_coordinates_np(boxes[:, 1], boxes[:, 3], h)
            if draw:
                image = self.draw(image, boxes, scores, classes, masks)
        return image, boxes, masks, classes, scores

    # 处理一张图片
    def eval_image(self, image, h, w, top_k=100):
        img = np.expand_dims(image, axis=0)

        feed_dic = {'input_1': img.astype(np.float32), 'priors': self.priors.astype(np.float32)}

        boxes, masks, classes, scores = self.exe.run(fluid.default_main_program(),
                                             feed=feed_dic,
                                             fetch_list=[self.boxes, self.masks, self.classes, self.scores])

        # 把增加的多余框去掉
        keep = np.where(scores > self._t1)[0]
        boxes = boxes[keep]
        scores = scores[keep]
        masks = masks[:, :, keep]
        classes = classes[keep]
        if len(keep) != 0:
            # 保留前k个
            boxes = boxes[:top_k, :]
            scores = scores[:top_k]
            masks = masks[:, :, :top_k]
            classes = classes[:top_k]

            masks = cv2.resize(masks, (w, h), interpolation=cv2.INTER_LINEAR)
            masks = np.reshape(masks, (h, w, -1))
            masks = masks.transpose(2, 0, 1)
            masks = (masks > 0.5).astype(np.float32)
            boxes[:, 0], boxes[:, 2] = sanitize_coordinates_np(boxes[:, 0], boxes[:, 2], w)
            boxes[:, 1], boxes[:, 3] = sanitize_coordinates_np(boxes[:, 1], boxes[:, 3], h)
        return boxes, masks, classes, scores

    def get_classes(self, file):
        with open(file) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]

        return class_names

    def draw(self, image, boxes, scores, classes, masks, mask_alpha=0.45):
        image_h, image_w, _ = image.shape
        # 定义颜色
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for box, score, cl, ms in zip(boxes, scores, classes, masks):
            bbox_color = colors[cl]

            # 在这里上掩码颜色
            color = np.array(bbox_color)
            color = np.reshape(color, (1, 1, 3))
            ms = np.expand_dims(ms, axis=2)
            ms = np.tile(ms, (1, 1, 3))
            color_ms = ms * color * mask_alpha
            color_im = ms * image * (1 - mask_alpha)
            image = color_im + color_ms + (1 - ms) * image

            # 画框
            x0, y0, x1, y1 = box
            left = max(0, np.floor(x0 + 0.5).astype(int))
            top = max(0, np.floor(y0 + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
            bbox_color = colors[cl]
            # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
            bbox_thick = 1
            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
            bbox_mess = '%s: %.2f' % (self.all_classes[cl], score)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        return image


