#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-02-01 19:27:23
#   Description : paddlepaddle_yolact++
#
# ================================================================
import math
import numpy as np


MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)

class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


def make_priors(max_size, conv_h, conv_w, aspect_ratios, scales, preapply_sqrt, use_pixel_scales, use_square_anchors):
    prior_data = []
    # Iteration order is important (it has to sync up with the convout)
    for j in range(conv_h):
        for i in range(conv_w):
            # +0.5 because priors are in center-size notation
            x = (i + 0.5) / conv_w
            y = (j + 0.5) / conv_h

            for ars in aspect_ratios:
                for scale in scales:
                    for ar in ars:
                        if not preapply_sqrt:
                            ar = math.sqrt(ar)
                        if use_pixel_scales:
                            w = scale * ar / max_size
                            h = scale / ar / max_size
                        else:
                            w = scale * ar / conv_w
                            h = scale / ar / conv_h

                        # This is for backward compatability with a bug where I made everything square by accident
                        if use_square_anchors:
                            h = w

                        prior_data += [x, y, w, h]
    return prior_data

def get_transform(backbone_name):
    if backbone_name[:6] == 'resnet':
        return Config({
                'channel_order': 'RGB',
                'normalize': True,
                'subtract_means': False,
                'to_float': False,
            })
    elif backbone_name == 'darknet53':
        return Config({
                'channel_order': 'RGB',
                'normalize': False,
                'subtract_means': False,
                'to_float': True,
            })


def get_priors(input_size, num_classes, strides, pred_aspect_ratios, pred_scales, preapply_sqrt, use_pixel_scales, use_square_anchors):
    priors = []
    for i in range(len(pred_aspect_ratios)):
        prior = make_priors(input_size, strides[2+i], strides[2+i], pred_aspect_ratios[i], pred_scales[i], preapply_sqrt, use_pixel_scales, use_square_anchors)
        priors += prior
    priors = np.array(priors)
    priors = np.reshape(priors, (-1, 4))
    class_vectors = np.identity(num_classes).astype(np.float32)
    num_priors_list = []
    for i in range(len(pred_aspect_ratios)):
        aspect_ratios = pred_aspect_ratios[i]
        scales = pred_scales[i]
        num_priors = sum(len(x) * len(scales) for x in aspect_ratios)
        num_priors_list.append(num_priors)
    num_priors = len(priors)
    return priors, class_vectors, num_priors, num_priors_list

def get_strides(input_size):
    a = input_size
    strides = []
    for i in range(1, 8, 1):
        if a % 2 == 0:
            a = a // 2
        else:
            a = (a + 1) // 2
        strides.append(a)
    return strides


