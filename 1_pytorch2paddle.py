#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-02-08 19:16:35
#   Description : 将yolact作者的pytorch模型转换为paddlepaddle模型。（地址https://github.com/dbolya/yolact）
#                 再对转换好的模型finetune，减少训练时间。
#                 唯一import torch的地方。
#                 这个脚本在本地windows上跑，再把pretrained_resnet50dcn模型上传到AIStudio。如果不允许，请从头训练。
#
# ================================================================
import torch
import paddle.fluid as fluid
from model.decode import Decode
import paddle.fluid.layers as P
from model.yolact import maskiou_net


def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))

    # For backward compatability, remove these (the new variable is called layers)
    for key in list(state_dict.keys()):
        if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
            del state_dict[key]
    return state_dict


# 为了maskiou_net权重
num_classes = 1 + 80
maskiou_net_input = P.data(name='maskiou_net_input', shape=[-1, 1, -1, -1], append_batch_size=False, dtype='float32')
maskiou_p = maskiou_net(maskiou_net_input, num_classes-1)

# 为了learning_rate_0
optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
optimizer.minimize(P.reduce_sum(maskiou_p))

# 建立模型。demo.py里的一部分
file = 'data/coco_classes.txt'
backbone_names = ['resnet50', 'resnet101', 'resnet50dcn', 'darknet53']
backbone_name = backbone_names[2
]
_decode = Decode(backbone_name, 550, 0.05, 0.5, model_path=None, file_path=file, use_gpu=False, is_test=False)
save_name = 'pretrained_' + backbone_name

# state_dict = load_weights('yolact_resnet50_54_800000.pth')
# state_dict = load_weights('yolact_base_54_800000.pth')
state_dict = load_weights('yolact_plus_resnet50_54_800000.pth')
# state_dict = load_weights('yolact_darknet53_54_800000.pth')


backbone_dic = {}
fpn = {}
proto_net = {}
prediction_layers = {}
semantic = {}
maskiou_net = {}
for key, value in state_dict.items():
    if 'tracked' in key:
        continue
    if 'backbone' in key:
        backbone_dic[key] = value.data.numpy()
    if 'fpn' in key:
        fpn[key] = value.data.numpy()
    if 'proto_net' in key:
        proto_net[key] = value.data.numpy()
    if 'prediction_layers' in key:
        prediction_layers[key] = value.data.numpy()
    if 'semantic' in key:
        semantic[key] = value.data.numpy()
    if 'maskiou_net' in key:
        maskiou_net[key] = value.data.numpy()


print('============================================================')


def copy1(k, prev, place):
    tensor = fluid.global_scope().find_var('conv%.2d.conv.weights' % k).get_tensor()
    tensor2 = fluid.global_scope().find_var('conv%.2d.bn.scale' % k).get_tensor()
    tensor3 = fluid.global_scope().find_var('conv%.2d.bn.offset' % k).get_tensor()
    tensor4 = fluid.global_scope().find_var('conv%.2d.bn.mean' % k).get_tensor()
    tensor5 = fluid.global_scope().find_var('conv%.2d.bn.var' % k).get_tensor()

    w = backbone_dic[prev + '.0.weight']
    y = backbone_dic[prev + '.1.weight']
    b = backbone_dic[prev + '.1.bias']
    m = backbone_dic[prev + '.1.running_mean']
    v = backbone_dic[prev + '.1.running_var']

    tensor.set(w, place)
    tensor2.set(y, place)
    tensor3.set(b, place)
    tensor4.set(m, place)
    tensor5.set(v, place)

def copy2(k, prev, n, place):
    i = 0
    for p in range(n):
        copy1(k + i + 1, prev + '.%d.conv1' % (p + 1), place)
        copy1(k + i + 2, prev + '.%d.conv2' % (p + 1), place)
        i += 2


def copy3(name, dic, place):
    v = fluid.global_scope().find_var(name)
    if v is None:
        print(name)
        return
    tensor = v.get_tensor()
    w = dic[name]
    tensor.set(w, place)

def copy_maskiou_net(name, dic, place):
    ss = name.split('.')
    name2 = name[len(ss[0])+1:]
    v = fluid.global_scope().find_var(name2)
    if v is None:
        print(name2)
        return
    tensor = v.get_tensor()
    w = dic[name]
    tensor.set(w, place)

# backbone_dic
if backbone_name == 'darknet53':
    copy1(1, 'backbone._preconv', _decode.place)
    copy1(2, 'backbone.layers.0.0', _decode.place)
    copy1(5, 'backbone.layers.1.0', _decode.place)
    copy1(10, 'backbone.layers.2.0', _decode.place)
    copy1(27, 'backbone.layers.3.0', _decode.place)
    copy1(44, 'backbone.layers.4.0', _decode.place)
    copy2(2, 'backbone.layers.0', 1, _decode.place)
    copy2(5, 'backbone.layers.1', 2, _decode.place)
    copy2(10, 'backbone.layers.2', 8, _decode.place)
    copy2(27, 'backbone.layers.3', 8, _decode.place)
    copy2(44, 'backbone.layers.4', 4, _decode.place)
else:
    for key, value in backbone_dic.items():
        copy3(key, backbone_dic, _decode.place)
# fpn
for key, value in fpn.items():
    copy3(key, fpn, _decode.place)
# proto_net
for key, value in proto_net.items():
    copy3(key, proto_net, _decode.place)
# prediction_layers
for key, value in prediction_layers.items():
    copy3(key, prediction_layers, _decode.place)
# semantic
for key, value in semantic.items():
    copy3(key, semantic, _decode.place)
# maskiou_net
for key, value in maskiou_net.items():
    copy_maskiou_net(key, maskiou_net, _decode.place)

fluid.io.save_persistables(_decode.exe, save_name, fluid.default_startup_program())
print('\nDone.')

