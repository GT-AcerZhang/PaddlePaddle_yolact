#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-01-23 15:16:15
#   Description : paddlepaddle_yolact++
#
# ================================================================
import cv2
import paddle.fluid as fluid
import paddle.fluid.layers as P
import sys
import argparse
import time
import shutil
import math
import numpy as np
import os

from data.coco import COCODetection
from model.losses import MultiBoxLoss
from model.yolact import Yolact
from utils.tools import get_transform, MEANS, STD, get_priors, get_strides
from utils import SSDAugmentation, BaseTransform


parser = argparse.ArgumentParser(description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
parser.add_argument('--initial_step', type=int,
                    help='Resume training at this step.')
parser.add_argument('--steps', default=900000, type=int,
                    help='Total train steps.')
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                    help='Initial learning rate.')
parser.add_argument('--pos_threshold', default=0.5, type=float,
                    help='Any of those priors whose maximum overlap is over the positive threshold, mark as positive.')
parser.add_argument('--neg_threshold', default=0.4, type=float,
                    help='For any priors whose maximum is less than the negative iou threshold, mark them as negative.The rest are neutral and not used in calculating the loss.')
parser.add_argument('--conf_loss', default='ce_loss',
                    help='select one of conf_loss from ce_loss, ghm_c_loss, focal_loss, ohem_loss.',
                    choices=['ce_loss', 'ghm_c_loss', 'focal_loss', 'ohem_loss'], type=str)
parser.add_argument('--negpos_ratio', default=3, type=int,
                    help='If you select ohem_loss as conf_loss, you need to assign the ratio between positives and negatives (3 means 3 negatives to 1 positive)')
parser.add_argument('--num_classes', default=81, type=int,
                    help='This should include the background class.')
parser.add_argument('--input_size', default=550, type=int,
                    help='Input size.')
parser.add_argument('--mask_dim', default=32, type=int,
                    help='The number of proto masks.')
parser.add_argument('--eval', default=False, type=bool,
                    help='Whether run evaluate dataset after each epoch.')
parser.add_argument('--pattern', type=int,
                    choices=[0, 1],
                    help='Training pattern. 0 means training from scratch. 1 means resume training from one model.')
parser.add_argument('--model_path', type=str, default='./pretrained_resnet50dcn',
                    help='If you select pattern=1, you need to assign the model_path.')
parser.add_argument('--use_gpu', default=True, type=bool)
parser.add_argument('--train_images_path', type=str, default='../data/data7122/train2017/')
parser.add_argument('--train_anno_path', type=str, default='../data/data7122/annotations/instances_train2017.json')
parser.add_argument('--valid_images_path', type=str, default='../data/data7122/val2017/')
parser.add_argument('--valid_anno_path', type=str, default='../data/data7122/annotations/instances_val2017.json')
parser.add_argument('--backbone_name', type=str, default='resnet50dcn',
                    choices=['resnet50', 'resnet101', 'resnet50dcn', 'darknet53'])
args = parser.parse_args()

batch_size = args.batch_size
initial_step = args.initial_step
steps = args.steps
lr = args.lr
pos_threshold = args.pos_threshold
neg_threshold = args.neg_threshold
use_ce_loss = False
use_ghm_c_loss = False
use_focal_loss = False
use_ohem_loss = False
if args.conf_loss == 'ce_loss':
    use_ce_loss = True
elif args.conf_loss == 'ghm_c_loss':
    use_ghm_c_loss = True
elif args.conf_loss == 'focal_loss':
    use_focal_loss = True
elif args.conf_loss == 'ohem_loss':
    use_ohem_loss = True
negpos_ratio = args.negpos_ratio
num_classes = args.num_classes
input_size = args.input_size
mask_dim = args.mask_dim  # 原型数
eval = args.eval    # 是否跑验证集
pattern = args.pattern   # 训练模式。 0-从头训练，1-读取模型继续训练
use_gpu = args.use_gpu
model_path = args.model_path
train_images_path = args.train_images_path
train_anno_path = args.train_anno_path
valid_images_path = args.valid_images_path
valid_anno_path = args.valid_anno_path
backbone_name = args.backbone_name

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def point_form_np(boxes):
    # (cx, cy, w, h)变成(x0, y0, x1, y1)
    return np.concatenate([boxes[:, :2] - boxes[:, 2:] * 0.5,
                           boxes[:, :2] + boxes[:, 2:] * 0.5], axis=-1)


def intersect_np(box_a, box_b):   # 相交区域的面积
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
    n = np.shape(box_a)[0]
    A = np.shape(box_a)[1]
    B = np.shape(box_b)[1]

    box_a = np.reshape(box_a, (n, A, 1, 4))
    box_b = np.reshape(box_b, (n, 1, B, 4))
    expand_box_a = np.repeat(box_a, B, axis=2)
    expand_box_b = np.repeat(box_b, A, axis=1)

    # 相交矩形的左上角坐标、右下角坐标
    left_up = np.maximum(expand_box_a[:, :, :, :2], expand_box_b[:, :, :, :2])
    right_down = np.minimum(expand_box_a[:, :, :, 2:], expand_box_b[:, :, :, 2:])

    inter_section = np.maximum(right_down - left_up, np.zeros(np.shape(left_up)))
    return inter_section[:, :, :, 0] * inter_section[:, :, :, 1]


def jaccard_np(box_a, box_b, iscrowd: bool = False):
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
        box_a = np.reshape(box_a, (1, np.shape(box_a)[0], np.shape(box_a)[1]))
        box_b = np.reshape(box_b, (1, np.shape(box_b)[0], np.shape(box_b)[1]))

    inter = intersect_np(box_a, box_b)

    area_a = (box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])
    area_a = np.reshape(area_a, (np.shape(area_a)[0], np.shape(area_a)[1], 1))
    area_a = np.repeat(area_a, np.shape(inter)[2], axis=2)

    area_b = (box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])
    area_b = np.reshape(area_b, (np.shape(area_b)[0], 1, np.shape(area_b)[1]))
    area_b = np.repeat(area_b, np.shape(inter)[1], axis=1)

    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out[0]

def center_size_np(boxes):
    # (x0, y0, x1, y1)变成(cx, cy, w, h)
    return np.concatenate([(boxes[:, :2] + boxes[:, 2:]) * 0.5,
                           boxes[:, 2:] - boxes[:, :2]], axis=-1)

def encode_np(matched, priors, use_yolo_regressors: bool = False):
    """ 编码。坐标解码的逆过程。label的填写。默认用的是SSD的编码方式 """
    if use_yolo_regressors:
        # Exactly the reverse of what we did in decode
        # In fact encode(decode(x, p), p) should be x
        boxes = center_size_np(matched)

        loc = np.concatenate([boxes[:, :2] - priors[:, :2],
                              np.log(boxes[:, 2:] / priors[:, 2:])], axis=-1)
    else:
        variances = [0.1, 0.2]

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = np.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        loc = np.concatenate([g_cxcy, g_wh], axis=-1)  # [num_priors,4]
    return loc


def match(pos_thresh, neg_thresh, labels_best_truth_idx, label_x0y0x1y1, priors,
          label_cid, label_crowd_x0y0x1y1, labels_pos_mask, labels_neg_mask, labels_vector, labels_pos_index, labels_pos_cid, labels_pos_cid2, idx, crowd_iou_threshold=0.7,
          use_yolo_regressors=False):
    decoded_priors = point_form_np(priors)

    # Size [num_objects, num_priors]
    overlaps = jaccard_np(label_x0y0x1y1, decoded_priors)

    # Size [num_priors] best ground truth for each prior
    best_truth_overlap = np.max(overlaps, axis=0)
    best_truth_idx = np.argmax(overlaps, axis=0)

    # We want to ensure that each gt gets used at least once so that we don't
    # waste any training data. In order to do that, find the max overlap anchor
    # with each gt, and force that anchor to use that gt.
    for _ in range(np.shape(overlaps)[0]):
        # Find j, the gt with the highest overlap with a prior
        # In effect, this will loop through overlaps.size(0) in a "smart" order,
        # always choosing the highest overlap first.
        best_prior_overlap = np.max(overlaps, axis=1)
        best_prior_idx = np.argmax(overlaps, axis=1)
        j = np.argmax(best_prior_overlap, axis=0)

        # Find i, the highest overlap anchor with this gt
        i = best_prior_idx[j]

        # Set all other overlaps with i to be -1 so that no other gt uses it
        overlaps[:, i] = -1
        # Set all other overlaps with j to be -1 so that this loop never uses j again
        overlaps[j, :] = -1

        # Overwrite i's score to be 2 so it doesn't get thresholded ever
        best_truth_overlap[i] = 2
        # Set the gt to be used for i to be j, overwriting whatever was there
        best_truth_idx[i] = j

    pos_label_cid = label_cid[best_truth_idx] + 1  # Shape: [num_priors]

    pos_label_cid2 = np.copy(pos_label_cid)
    pos_label_cid2[best_truth_overlap < pos_thresh] = 0  # used for focal_loss

    pos_label_cid[best_truth_overlap < pos_thresh] = -1  # label as neutral
    pos_label_cid[best_truth_overlap < neg_thresh] = 0  # label as background

    # Deal with crowd annotations for COCO
    if label_crowd_x0y0x1y1 is not None and crowd_iou_threshold < 1:
        # Size [num_priors, num_crowds]
        crowd_overlaps = jaccard_np(decoded_priors, label_crowd_x0y0x1y1, iscrowd=True)
        # Size [num_priors]
        best_crowd_overlap = np.max(crowd_overlaps, axis=1)
        # Set non-positives with crowd iou of over the threshold to be neutral.
        # 在反例中选择best_crowd_overlap > crowd_iou_threshold的作为忽略
        pos_label_cid[(pos_label_cid <= 0) & (best_crowd_overlap > crowd_iou_threshold)] = -1

    # 正例掩码。
    pos_mask = (pos_label_cid > 0).astype(np.float32)   # Shape: [19248, ]
    # 反例掩码。
    neg_mask = (pos_label_cid == 0).astype(np.float32)   # Shape: [19248, ]

    pos_label_x0y0x1y1 = label_x0y0x1y1[best_truth_idx]   # Shape: [19248, 4]
    pos_encode_x0y0x1y1 = encode_np(pos_label_x0y0x1y1, priors, use_yolo_regressors)   # Shape: [19248, 4]

    pos_index = np.where(pos_mask > 0.0)[0]
    pos_mask = np.reshape(pos_mask, (-1, 1))     # Shape: [19248, 1]
    neg_mask = np.reshape(neg_mask, (-1, 1))     # Shape: [19248, 1]
    best_truth_idx = np.reshape(best_truth_idx, (-1, 1))     # Shape: [19248, 1]
    pos_index = np.reshape(pos_index, (-1, 1))   # Shape: [?, 1]

    labels_pos_mask[idx, :, :] = pos_mask  # Shape: [batch_size, 19248, 1]
    labels_neg_mask[idx, :, :] = neg_mask  # Shape: [batch_size, 19248, 1]
    labels_vector[idx, :, 4:8] = pos_label_x0y0x1y1
    labels_vector[idx, :, 0:4] = pos_encode_x0y0x1y1
    labels_best_truth_idx[idx, :, :] = best_truth_idx
    labels_pos_index[idx] = np.copy(pos_index)
    labels_pos_cid[idx, :] = np.copy(pos_label_cid)
    labels_pos_cid2[idx, :] = np.copy(pos_label_cid2)

def generate_one_batch(dataset, indexes, step, batch_size, priors, strides, num_classes, num_priors, class_vectors):
    n = len(indexes)
    feed_dic = {'priors': priors.astype(np.float32), 'class_vectors': class_vectors}


    imgs = [None] * batch_size
    label_x0y0x1y1cid = [None] * batch_size
    label_masks = [None] * batch_size
    label_num_crowds = [None] * batch_size
    label_cid = [None] * batch_size

    labels_pos_mask = np.zeros((batch_size, num_priors, 1))
    labels_neg_mask = np.zeros((batch_size, num_priors, 1))
    labels_vector = np.zeros((batch_size, num_priors, 8))
    labels_pos_index = [None] * batch_size   # 正例下标
    labels_pos_cid = np.zeros((batch_size, num_priors))
    labels_pos_cid2 = np.zeros((batch_size, num_priors))

    labels_best_truth_idx = np.zeros((batch_size, num_priors, 1))


    if (step+1)*batch_size > n:
        batch = indexes[n-batch_size:n]
    else:
        batch = indexes[step*batch_size:(step+1)*batch_size]
    for idx in range(batch_size):
        imgs[idx], label_x0y0x1y1cid[idx], label_masks[idx], label_num_crowds[idx] = dataset.pull_item(batch[idx])

        imgs[idx] = np.reshape(imgs[idx], (1, )+np.shape(imgs[idx]))

        label_x0y0x1y1 = label_x0y0x1y1cid[idx][:, :-1]  # 坐标
        label_cid[idx] = label_x0y0x1y1cid[idx][:, -1].astype(np.int32)  # 类别id

        # Split the crowd annotations because they come bundled in
        cur_crowds = label_num_crowds[idx]
        if cur_crowds > 0:
            split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
            crowd_boxes, label_x0y0x1y1 = split(label_x0y0x1y1)  # 切分truths为 群体标记的boxes 和 单体标记的boxes
            # 我们丢弃群体标记的 类别id 和 masks (即丢弃变量_)
            _, label_cid[idx] = split(label_cid[idx])
            _, label_masks[idx] = split(label_masks[idx])
        else:
            crowd_boxes = None
        downsampled_masks_s8 = cv2.resize(label_masks[idx].transpose(1, 2, 0), (strides[2], strides[2]),
                                          interpolation=cv2.INTER_LINEAR)
        downsampled_masks_s8 = np.reshape(downsampled_masks_s8, (strides[2], strides[2], -1))
        downsampled_masks_s8 = downsampled_masks_s8.transpose(2, 0, 1)
        downsampled_masks_s8 = (downsampled_masks_s8 > 0.5).astype(np.float32)
        segment_t = np.zeros((num_classes - 1, strides[2], strides[2]))
        for obj_idx in range(np.shape(downsampled_masks_s8)[0]):
            segment_t[label_cid[idx][obj_idx]] = np.maximum(segment_t[label_cid[idx][obj_idx]],
                                                            downsampled_masks_s8[obj_idx])
        feed_dic['segment_t_%.2d' % idx] = segment_t.astype(np.float32)
        downsampled_masks_s4 = cv2.resize(label_masks[idx].transpose(1, 2, 0), (strides[1], strides[1]),
                                          interpolation=cv2.INTER_LINEAR)
        downsampled_masks_s4 = np.reshape(downsampled_masks_s4, (strides[1], strides[1], -1))
        downsampled_masks_s4 = downsampled_masks_s4.transpose(2, 0, 1)
        downsampled_masks_s4 = (downsampled_masks_s4 > 0.5).astype(np.float32)
        feed_dic['label_masks_%.2d' % idx] = downsampled_masks_s4.astype(np.float32)

        match(
            pos_threshold, neg_threshold, labels_best_truth_idx,
            label_x0y0x1y1, priors, label_cid[idx], crowd_boxes, labels_pos_mask, labels_neg_mask, labels_vector, labels_pos_index, labels_pos_cid, labels_pos_cid2, idx)
        feed_dic['labels_pos_index_%.2d' % idx] = labels_pos_index[idx].astype(np.int32)
    feed_dic['labels_pos_cid'] = labels_pos_cid.astype(np.int32)
    feed_dic['labels_pos_cid2'] = labels_pos_cid2.astype(np.int32)
    feed_dic['labels_best_truth_idx'] = labels_best_truth_idx.astype(np.int32)
    feed_dic['labels_pos_mask'] = labels_pos_mask.astype(np.float32)
    feed_dic['labels_neg_mask'] = labels_neg_mask.astype(np.float32)
    feed_dic['labels_vector'] = labels_vector.astype(np.float32)

    feed_dic['input_1'] = np.concatenate(imgs, axis=0).astype(np.float32)
    return feed_dic


if __name__ == '__main__':
    import platform
    sysstr = platform.system()
    if sysstr == 'Windows':
        use_gpu = False   # 如果自己的Windows电脑没有gpu，就不用动
        # use_gpu = True   # 如果自己的Windows电脑有gpu，就解除注释

    # 配置
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
    priors, class_vectors, num_priors, num_priors_list = get_priors(input_size, num_classes, strides,
               pred_aspect_ratios, pred_scales, preapply_sqrt, use_pixel_scales,
               use_square_anchors)

    train_program = fluid.Program()
    startup_program = fluid.Program()
    test_program = None
    with fluid.program_guard(train_program, startup_program):
        inputs = P.data(name='input_1', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
        pred_outs = Yolact(backbone_name, inputs, num_classes, mask_dim, num_priors_list, is_test=False)
        # 建立损失函数
        priors_tensor = P.data(name='priors', shape=[-1, 4], append_batch_size=False, dtype='float32')  # 第一维和批大小无关，是19248
        class_vectors_tensor = P.data(name='class_vectors', shape=[-1, -1], append_batch_size=False,
                                      dtype='float32')  # 类别向量

        labels_pos_cid_tensor = P.data(name='labels_pos_cid', shape=[-1, -1], append_batch_size=False, dtype='int32')
        labels_pos_cid2_tensor = P.data(name='labels_pos_cid2', shape=[-1, -1], append_batch_size=False, dtype='int32')   # focal_loss
        labels_pos_index_tensor = []
        label_masks_tensor = []
        labels_best_truth_idx_tensor = P.data(name='labels_best_truth_idx', shape=[-1, num_priors, 1],
                                              append_batch_size=False, dtype='int32')  # Shape: [batch_size, 19248, 1]
        labels_pos_mask_tensor = P.data(name='labels_pos_mask', shape=[-1, num_priors, 1], append_batch_size=False,
                                        dtype='float32')  # Shape: [batch_size, 19248, 1]
        labels_neg_mask_tensor = P.data(name='labels_neg_mask', shape=[-1, num_priors, 1], append_batch_size=False,
                                        dtype='float32')  # Shape: [batch_size, 19248, 1]
        labels_allboxes_vector_tensor = P.data(name='labels_vector', shape=[-1, num_priors, 8], append_batch_size=False,
                                               dtype='float32')
        segment_t_tensor = []

        for idx in range(batch_size):
            segment_t_tensor.append(P.data(name='segment_t_%.2d' % idx, shape=[-1, -1, -1], append_batch_size=False, dtype='float32'))
            labels_pos_index_tensor.append(P.data(name='labels_pos_index_%.2d' % idx, shape=[-1, 1], append_batch_size=False, dtype='int32'))
            label_masks_tensor.append(P.data(name='label_masks_%.2d' % idx, shape=[-1, -1, -1], append_batch_size=False, dtype='float32'))

        loss_layer = MultiBoxLoss(num_classes, pos_threshold, neg_threshold, negpos_ratio)
        losses, loss = loss_layer(pred_outs,
                 labels_pos_mask_tensor,     # Shape: [batch_size, 19248, 1]
                 labels_neg_mask_tensor,     # Shape: [batch_size, 19248, 1]
                 labels_allboxes_vector_tensor,     # Shape: [batch_size, 19248, 8]
                 segment_t_tensor,           # list  Shape: [batch_size, 19248, 1]
                 label_masks_tensor,
                 labels_best_truth_idx_tensor,
                 labels_pos_index_tensor,
                 labels_pos_cid_tensor,
                 labels_pos_cid2_tensor,
                 priors_tensor,
                 class_vectors_tensor,
                 batch_size,
                 use_ce_loss=use_ce_loss,
                 use_ghm_c_loss=use_ghm_c_loss,
                 use_focal_loss=use_focal_loss,
                 use_ohem_loss=use_ohem_loss)
        loss.persistable = True

        # 在使用Optimizer之前，将train_program复制成一个test_program。之后使用测试数据运行test_program，就可以做到运行测试程序，而不影响训练结果。
        test_program = train_program.clone(for_test=True)

        # 写完网络和损失，要紧接着写优化器
        optimizer = fluid.optimizer.SGD(learning_rate=lr)
        optimizer.minimize(loss)

    # 参数随机初始化
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)

    if pattern == 1:
        fluid.io.load_persistables(exe, model_path, main_program=startup_program)
        # 上一行代码，设置的学习率会被覆盖，所以这里重新设置学习率。
        v = fluid.global_scope().find_var('learning_rate_0')
        _tensor = v.get_tensor()
        _tensor.set(np.array([lr]), place)
    elif pattern == 0:
        # 从头训练。若使用focal_loss，预测的概率初始化为0.01
        if use_focal_loss:
            prior = 0.01
            b = -math.log((1.0 - prior) / prior)
            c_w = fluid.global_scope().find_var('prediction_layers.0.conf_layer.weight').get_tensor()
            c_b = fluid.global_scope().find_var('prediction_layers.0.conf_layer.bias').get_tensor()
            src_w = np.array(c_w)
            c_w.set((np.zeros(src_w.shape)).astype(np.float32), place)
            src_b = np.array(c_b)
            c_b.set((np.ones(src_b.shape) * b).astype(np.float32), place)

    # 验证集和训练集
    dataset = COCODetection(image_path=train_images_path,
                            info_file=train_anno_path,
                            transform=SSDAugmentation(transform, input_size, MEANS, STD))
    if eval:
        val_dataset = COCODetection(image_path=valid_images_path,
                                    info_file=valid_anno_path,
                                    transform=BaseTransform(transform, input_size, MEANS, STD))  # 不使用数据增强

    num_train = len(dataset)
    if eval: num_val = len(val_dataset)


    # 一轮的步数
    train_steps = int(num_train / batch_size)
    if eval: val_steps = int(num_val / batch_size)

    # 轮数
    num_epochs = math.ceil(steps / train_steps)

    # 模型存放
    if not os.path.exists('weights/'):
        os.mkdir('weights/')

    best_val_loss = 0.0
    iteration = initial_step
    for t in range(num_epochs):
        if (t+1)*train_steps < iteration:
            continue
        print('Epoch %d/%d\n'%(t+1, num_epochs))
        epochStartTime = time.time()
        start = time.time()
        # 每个epoch之前洗乱
        train_indexes = np.arange(num_train)
        np.random.shuffle(train_indexes)
        train_epoch_loss, val_epoch_loss = [], []

        # 训练阶段
        for step in range(train_steps):
            feed_dic = generate_one_batch(dataset, train_indexes, step, batch_size, priors, strides, num_classes, num_priors, class_vectors)

            train_step_loss, = exe.run(train_program, feed=feed_dic, fetch_list=[loss.name])
            train_epoch_loss.append(train_step_loss)

            # 自定义进度条
            percent = ((step + 1) / train_steps) * 100
            num = int(29 * percent / 100)
            ETA = int((time.time() - epochStartTime) * (100 - percent) / percent)
            sys.stdout.write('\r{0}'.format(' ' * (len(str(train_steps)) - len(str(step + 1)))) + \
                             '{0}/{1} [{2}>'.format(step + 1, train_steps, '=' * num) + '{0}'.format('.' * (29 - num)) + ']' + \
                             ' - ETA: ' + str(ETA) + 's' + ' - loss: %.4f'%(train_step_loss, ))
            sys.stdout.flush()

            iteration += 1
            if iteration % 100 == 0:
                fluid.io.save_persistables(exe,
                                           './weights/step%.7d-ep%.3d-loss%.3f.pd' % (iteration, (t + 1), np.mean(train_epoch_loss)),
                                           train_program)

                path_dir = os.listdir('./weights/')
                eps = []
                names = []
                for name in path_dir:
                    if name[len(name) - 2:len(name)] == 'pd' and name[0:2] == 'st':
                        sss = name.split('-')
                        ep = int(sss[0][4:])
                        eps.append(ep)
                        names.append(name)
                if len(eps) >= 16:
                    i2 = eps.index(min(eps))
                    shutil.rmtree('./weights/' + names[i2])
            # 跑完 中断轮 剩下的步step时，这个epoch结束
            if iteration == (t+1)*train_steps:
                break
            # 到达最大steps时，训练结束
            if iteration >= steps:
                break

        # 验证阶段
        if eval:
            val_indexes = np.arange(num_val)
            for step in range(val_steps):
                feed_dic = generate_one_batch(val_dataset, val_indexes, step, batch_size, priors, strides, num_classes, num_priors, class_vectors)

                val_step_loss, = exe.run(test_program, feed=feed_dic, fetch_list=[loss.name])
                val_epoch_loss.append(val_step_loss)
        train_epoch_loss, val_epoch_loss = np.mean(train_epoch_loss), np.mean(val_epoch_loss)

        # 打印本轮训练结果
        if eval:
            sys.stdout.write(
                '\r{0}/{1} [{2}='.format(train_steps, train_steps, '=' * num) + '{0}'.format('.' * (29 - num)) + ']' + \
                ' - %ds' % (int(time.time() - epochStartTime),) + ' - loss: %.4f'%(train_epoch_loss, ) + ' - val_loss: %.4f\n'%(val_epoch_loss, ))
        else:
            sys.stdout.write(
                '\r{0}/{1} [{2}='.format(train_steps, train_steps, '=' * num) + '{0}'.format('.' * (29 - num)) + ']' + \
                ' - %ds' % (int(time.time() - epochStartTime),) + ' - loss: %.4f'%(train_epoch_loss, ) + '\n')
        sys.stdout.flush()

