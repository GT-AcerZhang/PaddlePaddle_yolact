#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-01-23 15:16:15
#   Description : paddlepaddle_yolact++
#
# ================================================================
import paddle.fluid as fluid
import paddle.fluid.layers as P
import numpy as np
from paddle.fluid.param_attr import ParamAttr
from model.resnet import Resnet50, Resnet101
from model.darknet53 import DarkNet53
from utils.tools import MEANS, STD

def fast_preprocess_layer(img, input_size, normalize, subtract_means, to_float, mean=MEANS, std=STD):
    ''' 对图片预处理。用paddle而不使用numpy来得到更快的速度。预测时使用。 '''

    # NCHW
    img = P.transpose(img, perm=[0, 3, 1, 2])
    img = P.image_resize(img, out_shape=[input_size, input_size], resample="BILINEAR")

    if normalize:
        m = P.create_tensor(dtype='float32')
        P.assign(np.array(mean).astype(np.float32), m)
        m = P.reshape(m, (1, 3, 1, 1))
        m = P.expand_as(m, target_tensor=img)
        v = P.create_tensor(dtype='float32')
        P.assign(np.array(std).astype(np.float32), v)
        v = P.reshape(v, (1, 3, 1, 1))
        v = P.expand_as(v, target_tensor=img)
        img = (img - m) / v
    elif subtract_means:
        m = P.create_tensor(dtype='float32')
        P.assign(np.array(mean).astype(np.float32), m)
        m = P.reshape(m, (1, 3, 1, 1))
        m = P.expand_as(m, target_tensor=img)
        img = (img - m)
    elif to_float:  # 只是归一化
        img = img / 255

    # 换成RGB格式
    img_rgb = P.concat([img[:, 2:3, :, :], img[:, 1:2, :, :], img[:, 0:1, :, :]], axis=1)

    # Return value is in channel order [n, c, h, w] and RGB
    return img_rgb

def FPN(s8, s16, s32):
    # y1
    y1 = P.conv2d(s32, 256, filter_size=(1, 1),
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="fpn.lat_layers.0.weight"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="fpn.lat_layers.0.bias"))

    # y2
    h_m, w_m = P.shape(s16)[2], P.shape(s16)[3]
    x = P.image_resize(y1, out_shape=[h_m, w_m], resample="BILINEAR")
    y2 = P.conv2d(s16, 256, filter_size=(1, 1),
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="fpn.lat_layers.1.weight"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="fpn.lat_layers.1.bias"))
    y2 = P.elementwise_add(x, y2, act=None)

    # y3
    h_s, w_s = P.shape(s8)[2], P.shape(s8)[3]
    x = P.image_resize(y2, out_shape=[h_s, w_s], resample="BILINEAR")
    y3 = P.conv2d(s8, 256, filter_size=(1, 1),
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="fpn.lat_layers.2.weight"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="fpn.lat_layers.2.bias"))
    y3 = P.elementwise_add(x, y3, act=None)

    # pred
    y1 = P.conv2d(y1, 256, filter_size=(3, 3), padding=1,
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="fpn.pred_layers.0.weight"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="fpn.pred_layers.0.bias"))
    y1 = P.relu(y1)
    y2 = P.conv2d(y2, 256, filter_size=(3, 3), padding=1,
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="fpn.pred_layers.1.weight"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="fpn.pred_layers.1.bias"))
    y2 = P.relu(y2)
    y3 = P.conv2d(y3, 256, filter_size=(3, 3), padding=1,
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="fpn.pred_layers.2.weight"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="fpn.pred_layers.2.bias"))
    y3 = P.relu(y3)

    # 再对y1下采样2次
    s64 = P.conv2d(y1, 256, filter_size=(3, 3), stride=2, padding=1,
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="fpn.downsample_layers.0.weight"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="fpn.downsample_layers.0.bias"))
    s128 = P.conv2d(s64, 256, filter_size=(3, 3), stride=2, padding=1,
                  param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="fpn.downsample_layers.1.weight"),
                  bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="fpn.downsample_layers.1.bias"))
    return y3, y2, y1, s64, s128

def proto_net(x):
    x = P.conv2d(x, 256, filter_size=(3, 3), stride=1, padding=1,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="proto_net.0.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="proto_net.0.bias"))
    x = P.relu(x)

    x = P.conv2d(x, 256, filter_size=(3, 3), stride=1, padding=1,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="proto_net.2.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="proto_net.2.bias"))
    x = P.relu(x)

    x = P.conv2d(x, 256, filter_size=(3, 3), stride=1, padding=1,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="proto_net.4.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="proto_net.4.bias"))
    x = P.relu(x)

    x = P.resize_bilinear(x, scale=float(2))
    x = P.relu(x)

    x = P.conv2d(x, 256, filter_size=(3, 3), stride=1, padding=1,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="proto_net.8.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="proto_net.8.bias"))
    x = P.relu(x)

    x = P.conv2d(x, 32, filter_size=(1, 1), stride=1,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="proto_net.10.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="proto_net.10.bias"))
    return x

def maskiou_net(x, num_class):
    x = P.conv2d(x, 8, filter_size=(3, 3), stride=2, padding=1,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="maskiou_net.0.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="maskiou_net.0.bias"))
    x = P.relu(x)

    x = P.conv2d(x, 16, filter_size=(3, 3), stride=2, padding=1,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="maskiou_net.2.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="maskiou_net.2.bias"))
    x = P.relu(x)

    x = P.conv2d(x, 32, filter_size=(3, 3), stride=2, padding=1,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="maskiou_net.4.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="maskiou_net.4.bias"))
    x = P.relu(x)

    x = P.conv2d(x, 64, filter_size=(3, 3), stride=2, padding=1,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="maskiou_net.6.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="maskiou_net.6.bias"))
    x = P.relu(x)

    x = P.conv2d(x, 128, filter_size=(3, 3), stride=2, padding=1,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="maskiou_net.8.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="maskiou_net.8.bias"))
    x = P.relu(x)

    x = P.conv2d(x, num_class, filter_size=(1, 1), stride=1, padding=0,
                 param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="maskiou_net.10.weight"),
                 bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="maskiou_net.10.bias"))
    x = P.relu(x)
    return x

def PredictionModule(x,
                     num_priors,
                     num_classes,
                     mask_dim,
                     shared_conv_w,
                     shared_conv_b,
                     shared_bbox_w,
                     shared_bbox_b,
                     shared_conf_w,
                     shared_conf_b,
                     shared_mask_w,
                     shared_mask_b):
    '''
    改编自DSSD算法中的PredictionModule，改成了3x3卷积。3个分支分别预测bbox、conf、mask系数。
               x
             / | \
        bbox conf mask
    '''
    x = P.conv2d(x, 256, filter_size=(3, 3), stride=1, padding=1,
                 param_attr=shared_conv_w,
                 bias_attr=shared_conv_b)
    x = P.relu(x)

    bbox_x = x
    conf_x = x
    mask_x = x

    bbox = P.conv2d(bbox_x, num_priors * 4, filter_size=(3, 3), stride=1, padding=1,
                    param_attr=shared_bbox_w,
                    bias_attr=shared_bbox_b)
    bbox = P.transpose(bbox, perm=[0, 2, 3, 1])
    bbox = P.reshape(bbox, (P.shape(bbox)[0], -1, 4))

    conf = P.conv2d(conf_x, num_priors * num_classes, filter_size=(3, 3), stride=1, padding=1,
                    param_attr=shared_conf_w,
                    bias_attr=shared_conf_b)
    conf = P.transpose(conf, perm=[0, 2, 3, 1])
    conf = P.reshape(conf, (P.shape(conf)[0], -1, num_classes))

    mask = P.conv2d(mask_x, num_priors * mask_dim, filter_size=(3, 3), stride=1, padding=1,
                    param_attr=shared_mask_w,
                    bias_attr=shared_mask_b)
    mask = P.transpose(mask, perm=[0, 2, 3, 1])
    mask = P.reshape(mask, (P.shape(mask)[0], -1, mask_dim))
    mask = P.tanh(mask)

    preds = {'loc': bbox, 'conf': conf, 'mask': mask}
    return preds

def Yolact(backbone_name, inputs, num_classes, mask_dim, num_priors_list, is_test,
           transform=None, input_size=550, use_fast_prep=False):

    if use_fast_prep:
        inputs = fast_preprocess_layer(inputs, input_size, transform.normalize, transform.subtract_means, transform.to_float)

    # 冻结层时（即trainable=False），bn的均值、标准差也还是会变化，只有设置is_test=True才保证不变
    trainable = not is_test
    if backbone_name == 'darknet53':
        backbone_s8, backbone_s16, backbone_s32 = DarkNet53(inputs, is_test, trainable)
    elif backbone_name == 'resnet50':
        backbone_s8, backbone_s16, backbone_s32 = Resnet50(inputs, is_test, trainable, use_dcn=False)
    elif backbone_name == 'resnet101':
        backbone_s8, backbone_s16, backbone_s32 = Resnet101(inputs, is_test, trainable, use_dcn=False)
    elif backbone_name == 'resnet50dcn':
        backbone_s8, backbone_s16, backbone_s32 = Resnet50(inputs, is_test, trainable, use_dcn=True)

    s8, s16, s32, s64, s128 = FPN(backbone_s8, backbone_s16, backbone_s32)

    # 1.mask原型，默认有32个原型，即通道数是32
    proto_x = s8
    proto_out = proto_net(proto_x)
    proto_out = P.relu(proto_out)
    proto_out = P.transpose(proto_out, perm=[0, 2, 3, 1])

    # 2.预测头。第一个PredictionModule里的4个卷积层的参数被后面的PredictionModule共享
    shared_conv_w = ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="prediction_layers.0.upfeature.0.weight")
    shared_conv_b = ParamAttr(initializer=fluid.initializer.Constant(0.0), name="prediction_layers.0.upfeature.0.bias")
    shared_bbox_w = ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="prediction_layers.0.bbox_layer.weight")
    shared_bbox_b = ParamAttr(initializer=fluid.initializer.Constant(0.0), name="prediction_layers.0.bbox_layer.bias")
    shared_conf_w = ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="prediction_layers.0.conf_layer.weight")
    shared_conf_b = ParamAttr(initializer=fluid.initializer.Constant(0.0), name="prediction_layers.0.conf_layer.bias")
    shared_mask_w = ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="prediction_layers.0.mask_layer.weight")
    shared_mask_b = ParamAttr(initializer=fluid.initializer.Constant(0.0), name="prediction_layers.0.mask_layer.bias")

    pred_outs = {'loc': [], 'conf': [], 'mask': []}
    for i, tensor in enumerate([s8, s16, s32, s64, s128]):
        num_priors = num_priors_list[i]
        # 预测bbox、conf、mask
        preds = PredictionModule(tensor, num_priors, num_classes, mask_dim, shared_conv_w, shared_conv_b, shared_bbox_w, shared_bbox_b, shared_conf_w, shared_conf_b, shared_mask_w, shared_mask_b)
        for key, value in preds.items():
            pred_outs[key].append(value)

    for key, value in pred_outs.items():
        pred_outs[key] = P.concat(value, axis=1)
    pred_outs['proto'] = proto_out

    if is_test:   # 预测状态
        print('----test----')
        pred_outs['conf'] = P.softmax(pred_outs['conf'])
    else:   # 训练状态
        print('----train----')
        pred_outs['segm'] = P.conv2d(s8, num_classes-1, filter_size=(1, 1),
                                     param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name="semantic_seg_conv.weight"),
                                     bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), name="semantic_seg_conv.bias"))
    return pred_outs



