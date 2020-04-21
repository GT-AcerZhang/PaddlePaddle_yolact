#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-02-01 19:27:23
#   Description : paddlepaddle_yolact++
#
# ================================================================
import paddle.fluid.layers as P
import math

from model.decode import decode
from model.yolact import maskiou_net


def point_form(boxes):
    # (cx, cy, w, h)变成(x0, y0, x1, y1)
    return P.concat([boxes[:, :2] - boxes[:, 2:] * 0.5,
                     boxes[:, :2] + boxes[:, 2:] * 0.5], axis=-1)

def center_size(boxes):
    # (x0, y0, x1, y1)变成(cx, cy, w, h)
    return P.concat([(boxes[:, :2] + boxes[:, 2:]) * 0.5,
                     boxes[:, 2:] - boxes[:, :2]], axis=-1)


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

def log_sum_exp(x):
    """预测为背景的概率是(axx是神经网络的输出)
    p = e^(a00-max)/[e^(a00-max)+e^(a01-max)+...+e^(a80-max)]
    取对数
    lnp = a00-max-ln[e^(a00-max)+e^(a01-max)+...+e^(a80-max)]
    移项
    a00 = lnp + max + ln[e^(a00-max)+e^(a01-max)+...+e^(a80-max)]
    如果真的是背景类，标记p=1，所以
    a00 = max + ln[e^(a00-max)+e^(a01-max)+...+e^(a80-max)]
    神经网络的输出要尽量接近等号右边，才能预测为背景类。
    """
    x_max = P.reduce_max(x)
    return P.log(P.reduce_sum(P.exp(x - x_max), 1)) + x_max



class MultiBoxLoss(object):

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes

        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio

        # If you output a proto mask with this area, your l1 loss will be l1_alpha
        # Note that the area is relative (so 1 would be the entire image)
        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

    def __call__(self, predictions,
                 labels_pos_mask,     # Shape: [batch_size, 19248, 1]
                 labels_neg_mask,     # Shape: [batch_size, 19248, 1]
                 labels_allboxes_vector,     # Shape: [batch_size, 19248, 8]
                 segment_t,           # list  Shape: [batch_size, 19248, 1]
                 label_masks,
                 labels_best_truth_idx,
                 labels_pos_index,
                 labels_pos_cid,    #  Shape: [batch_size, 19248]
                 labels_pos_cid2,   #  Shape: [batch_size, 19248]
                 priors,
                 class_vectors,
                 batch_size, use_maskiou=True,
                 use_ce_loss=True, use_ghm_c_loss=False, use_focal_loss=False, use_ohem_loss=False):

        pred_allboxes_encode_x0y0x1y1 = predictions['loc']   # Shape: [batch_size, 19248, 4]
        pred_allboxes_conf = predictions['conf']             # Shape: [batch_size, 19248, 1+80]
        pred_allboxes_mask_coef = predictions['mask']        # Shape: [batch_size, 19248, 原型数=32]
        pred_proto = predictions['proto']           # Shape: [batch_size, s4=138, s4=138, 原型数=32]
        pred_segm = predictions['segm']             # Shape: [batch_size, 类别数=80, s8=69, s8=69]

        labels_allboxes_x0y0x1y1 = labels_allboxes_vector[:, :, 0:4]       # Shape: [batch_size, 19248, 4]
        labels_allboxes_decode_x0y0x1y1 = labels_allboxes_vector[:, :, 4:8]       # Shape: [batch_size, 19248, 4]

        losses = {}

        # 1.bbox_loss，只有正例才计算。
        # bbox_alpha = 1.5
        # bbox_loss = P.smooth_l1(P.reshape(pred_allboxes_encode_x0y0x1y1, (-1, 4)), P.reshape(labels_allboxes_x0y0x1y1, (-1, 4)))
        # bbox_loss = P.reshape(labels_pos_mask, (-1, 1)) * bbox_loss
        # bbox_loss = P.reduce_sum(bbox_loss) * bbox_alpha
        # losses['B'] = bbox_loss

        # 1.bbox_loss，ciou_loss
        pred_x0y0x1y1 = []
        for idx in range(batch_size):
            temp = decode(pred_allboxes_encode_x0y0x1y1[idx], priors)
            pred_x0y0x1y1.append(temp)
        pred_x0y0x1y1 = P.concat(pred_x0y0x1y1, axis=0)   # Shape: [batch_size*num_priors, 4]
        pred_x0y0x1y1 = P.reshape(pred_x0y0x1y1, (batch_size, -1, 4))   # Shape: [batch_size, num_priors, 4]

        ciou = P.reshape(self.bbox_ciou(pred_x0y0x1y1, labels_allboxes_decode_x0y0x1y1), (batch_size, -1, 1))    # (batch_size, num_priors, 1)

        # 每个预测框ciou_loss的权重 = 2 - (ground truth的面积/图片面积)
        gt_area = (labels_allboxes_decode_x0y0x1y1[:, :, 2:3] - labels_allboxes_decode_x0y0x1y1[:, :, 0:1]) * \
                  (labels_allboxes_decode_x0y0x1y1[:, :, 3:4] - labels_allboxes_decode_x0y0x1y1[:, :, 1:2])
        bbox_loss_scale = 2.0 - gt_area
        ciou_loss = labels_pos_mask * bbox_loss_scale * (1 - ciou)
        bbox_alpha = 1.5
        ciou_loss = P.reduce_sum(ciou_loss) * bbox_alpha
        losses['B'] = ciou_loss


        # 2.mask_loss，只有正例才计算
        mask_h = P.shape(pred_proto)[1]
        mask_w = P.shape(pred_proto)[2]
        loss_m = 0
        maskiou_t_list = []
        maskiou_net_input_list = []
        label_t_list = []
        for idx in range(batch_size):
            # [[0], [0], [0], [0], [0], [0], [0], [0]]。把8个正样本的最匹配gt的下标（在label_x0y0x1y1cid[idx]中的下标）选出来。
            # 因为只有一个gt，所以下标全是0
            labels_pos_index[idx].stop_gradient = True
            cur_gt = P.gather(labels_best_truth_idx[idx], labels_pos_index[idx])    # (?, 1)
            cur_gt.stop_gradient = True
            cur_x0y0x1y1 = P.gather(labels_allboxes_decode_x0y0x1y1[idx], labels_pos_index[idx])    # (?, 4)

            proto_masks = pred_proto[idx]   # (138, 138, 32)
            # pred_mask_coef (batch_size, 19248, 32)。 把8个正样本预测的mask系数选出来。
            proto_coef = P.gather(pred_allboxes_mask_coef[idx], labels_pos_index[idx])   # (?, 32)

            # （?, 138, 138），把8个正样本所匹配的gt的真实mask抽出来。因为匹配到同一个gt，所以是同一个mask重复了8次。
            mask_t = P.gather(label_masks[idx], cur_gt)   # (?, 138, 138)
            # （?, ），把8个正样本所匹配的gt的真实cid抽出来。因为匹配到同一个gt，所以是同一个cid重复了8次。
            label_t = P.gather(labels_pos_cid[idx], labels_pos_index[idx])  # (?, )

            # Size: (138, 138, ?)  =  原型*系数转置
            pred_masks = P.matmul(proto_masks, proto_coef, transpose_y=True)
            pred_masks = P.sigmoid(pred_masks)   # sigmoid激活

            pred_masks = crop(pred_masks, cur_x0y0x1y1)
            pred_masks = P.transpose(pred_masks, perm=[2, 0, 1])

            masks_pos_loss = mask_t * (0 - P.log(pred_masks + 1e-9))            # 二值交叉熵，加了极小的常数防止nan
            masks_neg_loss = (1 - mask_t) * (0 - P.log(1 - pred_masks + 1e-9))  # 二值交叉熵，加了极小的常数防止nan
            pre_loss = (masks_pos_loss + masks_neg_loss)
            pre_loss = P.reduce_sum(pre_loss, dim=[1, 2])

            # gt面积越小，对应mask损失权重越大
            cur_cxcywh = center_size(cur_x0y0x1y1)
            gt_box_width = cur_cxcywh[:, 2]
            gt_box_height = cur_cxcywh[:, 3]
            pre_loss = pre_loss / (gt_box_width * gt_box_height)
            loss_m += P.reduce_sum(pre_loss)

            if use_maskiou:
                # mask_t中，面积<=5*5的被丢弃
                # discard_mask_area = 5*5
                '''
                gpu版本的paddlepaddle1.6.2里有一个问题。select如果是[None]，并且在gather()里使用了select，就会出现
                cudaGetLastError  invalid configuration argument errno: 9   这个错误。cpu版本则可以正常跑。
                为了避免上面的问题，只能让select不是[None]，所以这里不做面积过滤，mask_t全部保留。
                '''
                discard_mask_area = -1
                gt_mask_area = P.reduce_sum(mask_t, dim=[1, 2])
                gt_mask_area.stop_gradient = True
                select = P.where(gt_mask_area > discard_mask_area)
                select.stop_gradient = True
                pred_masks = P.gather(pred_masks, select)
                mask_t = P.gather(mask_t, select)
                label_t = P.gather(label_t, select)
                label_t.stop_gradient = True

                maskiou_net_input = P.reshape(pred_masks, (P.shape(pred_masks)[0], 1, mask_h, mask_w))
                pred_masks = P.cast(pred_masks > 0.5, 'float32')  # 四舍五入
                maskiou_t = self._mask_iou(pred_masks, mask_t)   # (8, )
                maskiou_net_input_list.append(maskiou_net_input) # (8, 1, 138, 138)
                maskiou_t_list.append(maskiou_t)   # (8, )
                label_t_list.append(label_t)       # (8, )
        mask_alpha = 6.125
        losses['M'] = loss_m * mask_alpha / mask_h / mask_w

        # 余下部分
        if use_maskiou:
            maskiou_net_input = P.concat(maskiou_net_input_list, axis=0) # (21, 1, 138, 138)  21个正例预测的掩码
            maskiou_t = P.concat(maskiou_t_list, axis=0)   # (21, )  21个正例预测的掩码和真实掩码的iou
            label_t = P.concat(label_t_list, axis=0)       # (21, )  21个正例预测的cid
            label_t.stop_gradient = True   # 因为是整数所以才？
            maskiou_targets = [maskiou_net_input, maskiou_t, label_t]


        # 3.conf_loss。
        conf_alpha = 1.0
        if use_ce_loss:
            conf_loss = self.ce_conf_loss(pred_allboxes_conf, labels_pos_mask, labels_neg_mask,
                                          class_vectors, labels_pos_cid2, gt_area)
        elif use_ghm_c_loss:
            conf_loss = self.ghm_c_loss(pred_allboxes_conf, labels_pos_mask, labels_neg_mask,
                                        class_vectors, labels_pos_cid2)
        elif use_focal_loss:
            conf_loss = self.focal_conf_loss(pred_allboxes_conf, labels_pos_mask, labels_neg_mask,
                                             class_vectors, labels_pos_cid2)
        elif use_ohem_loss:
            conf_loss = self.ohem_conf_loss(pred_allboxes_conf, batch_size,
                                            labels_neg_mask, labels_pos_mask,
                                            labels_pos_index, class_vectors, labels_pos_cid)
        losses['C'] = conf_loss * conf_alpha

        # 4.mask_iou_loss，只有正例才计算。
        if use_maskiou:
            # maskiou_net_input  (21, 1, 138, 138)  21个正例预测的掩码
            # maskiou_t          (21, )             21个正例预测的掩码和真实掩码的iou
            # label_t            (21, )             21个正例预测的cid
            maskiou_net_input, maskiou_t, label_t = maskiou_targets
            maskiou_p = maskiou_net(maskiou_net_input, self.num_classes-1)
            maskiou_p = P.reduce_max(maskiou_p, dim=[2, 3])   # 最大池化  (21, 80)
            temp_mask = P.gather(class_vectors, label_t)      # 掩码  (21, 81)
            temp_mask = temp_mask[:, 1:]                      # 掩码  (21, 80)
            maskiou_p = temp_mask * maskiou_p                 # 只保留真实类别的那个通道  (21, 80)
            maskiou_p = P.reduce_sum(maskiou_p, dim=1, keep_dim=True)        # (21, 1)
            loss_i = P.smooth_l1(maskiou_p, P.reshape(maskiou_t, (P.shape(maskiou_t)[0], 1)))
            maskiou_alpha = 25.0
            losses['I'] = maskiou_alpha * P.reduce_sum(loss_i)


        # 5.semantic_segmentation_loss，只有正例才计算
        mask_h = P.shape(pred_segm)[2]
        mask_w = P.shape(pred_segm)[3]
        loss_s = 0.0
        for idx in range(batch_size):
            cur_segment = pred_segm[idx]   # (80, 69, 69)
            l = P.sigmoid_cross_entropy_with_logits(cur_segment, segment_t[idx])
            loss_s += P.reduce_sum(l)

        semantic_segmentation_alpha = 1.0
        losses['S'] = loss_s / mask_h / mask_w * semantic_segmentation_alpha


        total_num_pos = P.cast(P.reduce_sum(labels_pos_mask), 'float32')
        for k in losses:
            if k not in ('S', ):
                losses[k] /= total_num_pos
            else:
                losses[k] /= batch_size
        total_loss = 0.0
        for k in losses:
            total_loss += losses[k]

        # Loss Key:
        #  - B: Box Localization Loss
        #  - M: Mask Loss
        #  - C: Class Confidence Loss
        #  - I: MaskIou Loss
        #  - S: Semantic Segmentation Loss
        # return losses['M'], losses['C']
        return losses, total_loss

    def _mask_iou(self, mask1, mask2):
        intersection = P.reduce_sum(mask1 * mask2, dim=[1, 2])
        area1 = P.reduce_sum(mask1, dim=[1, 2])
        area2 = P.reduce_sum(mask2, dim=[1, 2])
        union = (area1 + area2) - intersection
        ret = intersection / (union + 1e-9)
        return ret

    def ce_conf_loss(self, pred_allboxes_conf, labels_pos_mask, labels_neg_mask,
                     class_vectors, labels_pos_cid2,
                     gt_area):
        labels_pos_cid2 = P.reshape(labels_pos_cid2, (-1, ))  # [batch_size*num_priors]
        pred_allboxes_conf_r = P.reshape(pred_allboxes_conf, (-1, P.shape(pred_allboxes_conf)[2]))  # [batch_size*num_priors, num_classes]
        label_prob = P.gather(class_vectors, labels_pos_cid2)      # one-hot掩码  (batch_size*num_priors, num_classes)

        pred_prob = P.softmax(pred_allboxes_conf_r)
        pred_prob = P.cast(pred_prob, 'float32')
        prob_loss = label_prob * (0 - P.log(pred_prob + 1e-9))   # 加了极小的常数防止nan
        prob_loss = P.reduce_sum(prob_loss, dim=1)

        # 只留下正反例的损失
        labels_pos_mask2 = P.reshape(labels_pos_mask, (-1,))  # [batch_size*num_priors]
        labels_neg_mask2 = P.reshape(labels_neg_mask, (-1,))  # [batch_size*num_priors]
        conf_loss_scale = 2.0 - gt_area   # gt面积越小，权重越大，越受重视
        conf_loss_scale = P.reshape(conf_loss_scale, (-1,))  # [batch_size*num_priors]
        prob_pos_loss = prob_loss * labels_pos_mask2 * conf_loss_scale
        prob_neg_loss = prob_loss * labels_neg_mask2
        ce_loss = prob_pos_loss + prob_neg_loss
        ce_loss = P.reduce_sum(ce_loss)

        return ce_loss

    def ghm_c_loss(self, pred_allboxes_conf, labels_pos_mask, labels_neg_mask,
                   class_vectors, labels_pos_cid2):
        labels_pos_cid2 = P.reshape(labels_pos_cid2, (-1,))  # [batch_size*num_priors]
        pred_allboxes_conf_r = P.reshape(pred_allboxes_conf,
                                         (-1, P.shape(pred_allboxes_conf)[2]))  # [batch_size*num_priors, num_classes]
        label_prob = P.gather(class_vectors, labels_pos_cid2)  # one-hot掩码  (batch_size*num_priors, num_classes)

        # 我们可以在训练时改为sigmoid激活，预测时依然还是softmax激活。
        # 能这么做的原因是，若某位的sigmoid值最大，那么一定有该位的softmax值最大。
        pred_prob = P.sigmoid(pred_allboxes_conf_r)
        pred_prob = P.cast(pred_prob, 'float32')

        # 二值交叉熵损失，prob_neg_loss里其实含有忽略样本的损失，这部分不应该计算，后面会用掩码过滤。
        # 样本数量变成了batch_size*num_priors*num_classes，而不是batch_size*num_priors
        # 某个候选框（batch_size*num_priors个之一）若真实类别是7，那么7这个通道是正样本，该框余下80个通道是负样本
        # （负样本可不是指背景，而是与真实class_id通道不同的另外的通道的80个概率）

        # 梯度模长g。正样本是1-p，负样本是p
        pred_prob_copy = P.assign(pred_prob)
        g = (1 - pred_prob_copy) * label_prob + pred_prob_copy * (1 - label_prob)
        labels_pos_mask2 = P.reshape(labels_pos_mask, (-1,))  # [batch_size*num_priors]
        labels_neg_mask2 = P.reshape(labels_neg_mask, (-1,))  # [batch_size*num_priors]
        labels_pos_mask3 = P.reshape(labels_pos_mask, (-1, 1))  # [batch_size*num_priors, 1]
        labels_neg_mask3 = P.reshape(labels_neg_mask, (-1, 1))  # [batch_size*num_priors, 1]
        labels_pos_mask4 = P.expand_as(labels_pos_mask3, g)  # [batch_size*num_priors, num_classes]
        labels_neg_mask4 = P.expand_as(labels_neg_mask3, g)  # [batch_size*num_priors, num_classes]
        # 忽略样本(cid=-1)的g置-1.0
        g = g * (labels_pos_mask4 + labels_neg_mask4) + (-1.0) * (1 - labels_pos_mask4 - labels_neg_mask4)
        g.stop_gradient = True
        pred_prob.stop_gradient = False


        # g的取值范围[0, 1]划分为k个区域
        k = 5
        epsilon = 1.0 / k   # 区域长度
        w = 0
        c = P.cast(-0.5 <= g, 'float32') * P.cast(g < epsilon, 'float32')
        w += c * P.reduce_sum(c)
        for i in range(1, k-1, 1):
            c = P.cast(epsilon*i <= g, 'float32') * P.cast(g < epsilon*(i+1), 'float32')
            w += c * P.reduce_sum(c)
        c = P.cast(epsilon*(k-1) <= g, 'float32')
        w += c * P.reduce_sum(c)

        # 梯度密度
        GD = w * k

        # GHM_C_loss
        prob_pos_loss = label_prob * (0 - P.log(pred_prob + 1e-9)) / (GD + 1e-9)            # 加了极小的常数防止nan
        prob_neg_loss = (1 - label_prob) * (0 - P.log(1 - pred_prob + 1e-9)) / (GD + 1e-9)  # 加了极小的常数防止nan
        ghm_c_loss = prob_pos_loss + prob_neg_loss
        ghm_c_loss = P.reduce_sum(ghm_c_loss, dim=1)
        ghm_c_loss = ghm_c_loss * (labels_pos_mask2 + labels_neg_mask2)
        ghm_c_loss = P.reduce_sum(ghm_c_loss)

        return ghm_c_loss

    '''def focal_conf_loss(self, pred_allboxes_conf, labels_pos_mask, labels_neg_mask,
                        class_vectors, labels_pos_cid2,
                        focal_loss_alpha=0.25, focal_loss_gamma=2):
        labels_pos_cid2 = P.reshape(labels_pos_cid2, (-1, ))  # [batch_size*num_priors]
        pred_allboxes_conf_r = P.reshape(pred_allboxes_conf, (-1, P.shape(pred_allboxes_conf)[2]))  # [batch_size*num_priors, num_classes]
        label_prob = P.gather(class_vectors, labels_pos_cid2)      # one-hot掩码  (batch_size*num_priors, num_classes)

        pred_prob = P.softmax(pred_allboxes_conf_r)
        pred_prob = P.cast(pred_prob, 'float32')
        prob_loss = label_prob * (0 - P.log(pred_prob + 1e-9))   # 加了极小的常数防止nan
        prob_loss = P.reduce_sum(prob_loss, dim=1)

        # 只留下真实类别预测的概率
        pred_prob = label_prob * pred_prob
        pred_prob = P.reduce_sum(pred_prob, dim=1)

        # 加上正反例的权重
        labels_pos_mask2 = P.reshape(labels_pos_mask, (-1,))  # [batch_size*num_priors]
        labels_neg_mask2 = P.reshape(labels_neg_mask, (-1,))  # [batch_size*num_priors]
        prob_pos_loss = prob_loss * labels_pos_mask2 * focal_loss_alpha * (1 - pred_prob) ** focal_loss_gamma
        prob_neg_loss = prob_loss * labels_neg_mask2 * (1 - focal_loss_alpha) * pred_prob ** focal_loss_gamma
        focal_loss = prob_pos_loss + prob_neg_loss
        focal_loss = P.reduce_sum(focal_loss)

        return focal_loss'''

    def focal_conf_loss(self, pred_allboxes_conf, labels_pos_mask, labels_neg_mask,
                        class_vectors, labels_pos_cid2,
                        focal_loss_alpha=0.25, focal_loss_gamma=2):
        labels_pos_cid2 = P.reshape(labels_pos_cid2, (-1, ))  # [batch_size*num_priors]
        pred_allboxes_conf_r = P.reshape(pred_allboxes_conf, (-1, P.shape(pred_allboxes_conf)[2]))  # [batch_size*num_priors, num_classes]
        label_prob = P.gather(class_vectors, labels_pos_cid2)      # one-hot掩码  (batch_size*num_priors, num_classes)

        # 我们可以在训练时改为sigmoid激活，预测时依然还是softmax激活。
        # 能这么做的原因是，若某位的sigmoid值最大，那么一定有该位的softmax值最大。
        pred_prob = P.sigmoid(pred_allboxes_conf_r)
        pred_prob = P.cast(pred_prob, 'float32')

        # focal_loss
        labels_pos_mask2 = P.reshape(labels_pos_mask, (-1,))  # [batch_size*num_priors]
        labels_neg_mask2 = P.reshape(labels_neg_mask, (-1,))  # [batch_size*num_priors]
        prob_pos_loss = label_prob * (0 - P.log(pred_prob + 1e-9)) * focal_loss_alpha * (1.0 - pred_prob) ** focal_loss_gamma
        prob_neg_loss = (1 - label_prob) * (0 - P.log(1 - pred_prob + 1e-9)) * (1.0 - focal_loss_alpha) * pred_prob ** focal_loss_gamma
        focal_loss = prob_pos_loss + prob_neg_loss
        focal_loss = P.reduce_sum(focal_loss, dim=1)
        focal_loss = focal_loss * (labels_pos_mask2 + labels_neg_mask2)
        focal_loss = P.reduce_sum(focal_loss)

        return focal_loss

    def ohem_conf_loss(self, pred_allboxes_conf, batch_size, labels_neg_mask, labels_pos_mask,
                       labels_pos_index, class_vectors, labels_pos_cid):
        batch_conf = P.reshape(pred_allboxes_conf, (-1, self.num_classes))
        loss_c = log_sum_exp(batch_conf) - batch_conf[:, 0]
        loss_c = P.reshape(loss_c, (batch_size, -1))             # (batch_size, 19248)
        labels_neg_mask = P.concat(labels_neg_mask, axis=0)      # (batch_size*19248, 1)
        labels_neg_mask = P.reshape(labels_neg_mask, (batch_size, -1))     # (batch_size, 19248)
        loss_c = labels_neg_mask * loss_c             # 只留下负样本损失, (batch_size, 19248)
        sorted_loss_c, loss_idx = P.argsort(loss_c, axis=-1, descending=True)

        labels_pos_mask = P.concat(labels_pos_mask, axis=0)      # (batch_size*19248, 1)
        labels_pos_mask = P.reshape(labels_pos_mask, (batch_size, -1))     # (batch_size, 19248)
        num_pos = P.cast(P.reduce_sum(labels_pos_mask, dim=1), 'int32')    # (batch_size, )
        num_neg = self.negpos_ratio * num_pos          # (batch_size, )
        neg_topk_mask = []
        for idx in range(batch_size):
            desc = P.range(num_neg[idx], num_neg[idx]-P.shape(labels_pos_mask)[1], -1, 'int32')
            neg_topk_mask.append(desc)
        neg_topk_mask = P.concat(neg_topk_mask, axis=0)      # (batch_size*19248, )
        neg_topk_mask = P.reshape(neg_topk_mask, (batch_size, -1))     # (batch_size, 19248)
        neg_topk_mask = P.cast(neg_topk_mask > 0, 'float32')   # (batch_size, 19248)
        sorted_loss_c = neg_topk_mask * sorted_loss_c
        selected_poss = []
        selected_negs = []
        selected_pos_class_vectors = []
        selected_neg_class_vectors = []
        for idx in range(batch_size):
            selected_neg_idx_idx = P.where(sorted_loss_c[idx] > 0)
            selected_neg_idx_idx.stop_gradient = True
            selected_neg_idx = P.gather(loss_idx[idx], selected_neg_idx_idx)
            selected_neg_idx.stop_gradient = True
            selected_neg = P.gather(pred_allboxes_conf[idx], selected_neg_idx)
            selected_neg.stop_gradient = True
            selected_negs.append(selected_neg)
            selected_pos = P.gather(pred_allboxes_conf[idx], labels_pos_index[idx])
            selected_pos.stop_gradient = True
            selected_poss.append(selected_pos)

            zeros = P.fill_constant(shape=[P.shape(selected_neg)[0], ], value=0, dtype='int32')
            zeros.stop_gradient = True
            selected_neg_class_vector = P.gather(class_vectors, zeros)
            selected_neg_class_vector.stop_gradient = True
            selected_neg_class_vectors.append(selected_neg_class_vector)

            labels_pos_cid.stop_gradient = True
            labels_pos_index[idx].stop_gradient = True
            selected_pos_cid = P.gather(labels_pos_cid[idx], labels_pos_index[idx])
            selected_pos_cid.stop_gradient = True
            selected_pos_class_vector = P.gather(class_vectors, selected_pos_cid)
            selected_pos_class_vector.stop_gradient = True
            selected_pos_class_vectors.append(selected_pos_class_vector)
        selected_negs = P.concat(selected_negs, axis=0)      # (?, 1+80)
        selected_poss = P.concat(selected_poss, axis=0)      # (?, 1+80)
        pred_ = P.concat([selected_negs, selected_poss], axis=0)      # (?, 1+80)
        selected_neg_class_vectors = P.concat(selected_neg_class_vectors, axis=0)      # (?, 1+80)
        selected_pos_class_vectors = P.concat(selected_pos_class_vectors, axis=0)      # (?, 1+80)
        labels_ = P.concat([selected_neg_class_vectors, selected_pos_class_vectors], axis=0)      # (?, 1+80)

        # softmax交叉熵
        fenzi = P.exp(pred_)
        fenmu = P.reduce_sum(fenzi, dim=1, keep_dim=True)
        pred_prob = fenzi / P.expand_as(fenmu, target_tensor=fenzi)
        conf_loss = labels_ * (0 - P.log(pred_prob + 1e-9))  # 交叉熵，加了极小的常数防止nan
        conf_loss = P.reduce_sum(conf_loss)
        return conf_loss

    def bbox_ciou(self, boxes1_x0y0x1y1, boxes2_x0y0x1y1):
        '''
        计算ciou = iou - p2/c2 - av
        :param boxes1: (batch_size, num_priors, 4)   pred_x0y0x1y1
        :param boxes2: (batch_size, num_priors, 4)   label_x0y0x1y1
        :return:
        '''

        # 得到中心点坐标、宽高
        boxes1 = P.concat([(boxes1_x0y0x1y1[:, :, :2] + boxes1_x0y0x1y1[:, :, 2:]) * 0.5,
                           boxes1_x0y0x1y1[:, :, 2:] - boxes1_x0y0x1y1[:, :, :2]], axis=-1)
        boxes2 = P.concat([(boxes2_x0y0x1y1[:, :, :2] + boxes2_x0y0x1y1[:, :, 2:]) * 0.5,
                           boxes2_x0y0x1y1[:, :, 2:] - boxes2_x0y0x1y1[:, :, :2]], axis=-1)

        # 两个矩形的面积
        boxes1_area = (boxes1_x0y0x1y1[:, :, 2] - boxes1_x0y0x1y1[:, :, 0]) * (
                    boxes1_x0y0x1y1[:, :, 3] - boxes1_x0y0x1y1[:, :, 1])
        boxes2_area = (boxes2_x0y0x1y1[:, :, 2] - boxes2_x0y0x1y1[:, :, 0]) * (
                    boxes2_x0y0x1y1[:, :, 3] - boxes2_x0y0x1y1[:, :, 1])

        # 相交矩形的左上角坐标、右下角坐标
        left_up = P.elementwise_max(boxes1_x0y0x1y1[:, :, :2], boxes2_x0y0x1y1[:, :, :2])
        right_down = P.elementwise_min(boxes1_x0y0x1y1[:, :, 2:], boxes2_x0y0x1y1[:, :, 2:])

        # 相交矩形的面积inter_area。iou
        inter_section = P.relu(right_down - left_up)
        inter_area = inter_section[:, :, 0] * inter_section[:, :, 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        # 包围矩形的左上角坐标、右下角坐标
        enclose_left_up = P.elementwise_min(boxes1_x0y0x1y1[:, :, :2], boxes2_x0y0x1y1[:, :, :2])
        enclose_right_down = P.elementwise_max(boxes1_x0y0x1y1[:, :, 2:], boxes2_x0y0x1y1[:, :, 2:])

        # 包围矩形的对角线的平方
        enclose_wh = enclose_right_down - enclose_left_up
        enclose_c2 = P.pow(enclose_wh[:, :, 0], 2) + P.pow(enclose_wh[:, :, 1], 2)

        # 两矩形中心点距离的平方
        p2 = P.pow(boxes1[:, :, 0] - boxes2[:, :, 0], 2) + P.pow(boxes1[:, :, 1] - boxes2[:, :, 1], 2)

        # 增加av。分母boxes2[:, :, 3]可能为0，所以加了极小的常数防止nan
        atan1 = P.atan(boxes1[:, :, 2] / (boxes1[:, :, 3] + 1e-9))
        atan2 = P.atan(boxes2[:, :, 2] / (boxes2[:, :, 3] + 1e-9))
        v = 4.0 * P.pow(atan1 - atan2, 2) / (math.pi ** 2)
        a = v / (1 - iou + v)

        ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
        return ciou

