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
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

def conv2d_unit(x, filters, kernels, stride, padding, name, is_test, trainable):
    x = P.conv2d(
        input=x,
        num_filters=filters,
        filter_size=kernels,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.01), name=name + ".conv.weights", trainable=trainable),
        bias_attr=False)
    bn_name = name + ".bn"
    x = P.batch_norm(
        input=x,
        act=None,
        is_test=is_test,
        param_attr=ParamAttr(
            initializer=fluid.initializer.Constant(1.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=bn_name + '.scale'),
        bias_attr=ParamAttr(
            initializer=fluid.initializer.Constant(0.0),
            regularizer=L2Decay(0.),
            trainable=trainable,
            name=bn_name + '.offset'),
        moving_mean_name=bn_name + '.mean',
        moving_variance_name=bn_name + '.var')
    x = P.leaky_relu(x, alpha=0.1)
    return x

def residual_block(inputs, filters, conv_start_idx, is_test, trainable):
    x = conv2d_unit(inputs, filters, (1, 1), stride=1, padding=0, name='conv%.2d'% conv_start_idx, is_test=is_test, trainable=trainable)
    x = conv2d_unit(x, 2 * filters, (3, 3), stride=1, padding=1, name='conv%.2d'% (conv_start_idx+1), is_test=is_test, trainable=trainable)
    x = P.elementwise_add(x=inputs, y=x, act=None)
    return x

def stack_residual_block(inputs, filters, n, conv_start_idx, is_test, trainable):
    x = residual_block(inputs, filters, conv_start_idx, is_test, trainable)
    for i in range(n - 1):
        x = residual_block(x, filters, conv_start_idx+2*(1+i), is_test, trainable)
    return x

def DarkNet53(inputs, is_test, trainable):
    ''' 所有卷积层都没有偏移bias_attr=False '''
    x = conv2d_unit(inputs, 32, (3, 3), stride=1, padding=1, name='conv01', is_test=is_test, trainable=trainable)

    x = conv2d_unit(x, 64, (3, 3), stride=2, padding=1, name='conv02', is_test=is_test, trainable=trainable)
    x = stack_residual_block(x, 32, n=1, conv_start_idx=3, is_test=is_test, trainable=trainable)

    x = conv2d_unit(x, 128, (3, 3), stride=2, padding=1, name='conv05', is_test=is_test, trainable=trainable)
    x = stack_residual_block(x, 64, n=2, conv_start_idx=6, is_test=is_test, trainable=trainable)

    x = conv2d_unit(x, 256, (3, 3), stride=2, padding=1, name='conv10', is_test=is_test, trainable=trainable)
    s8 = stack_residual_block(x, 128, n=8, conv_start_idx=11, is_test=is_test, trainable=trainable)

    x = conv2d_unit(s8, 512, (3, 3), stride=2, padding=1, name='conv27', is_test=is_test, trainable=trainable)
    s16 = stack_residual_block(x, 256, n=8, conv_start_idx=28, is_test=is_test, trainable=trainable)

    x = conv2d_unit(s16, 1024, (3, 3), stride=2, padding=1, name='conv44', is_test=is_test, trainable=trainable)
    s32 = stack_residual_block(x, 512, n=4, conv_start_idx=45, is_test=is_test, trainable=trainable)
    return s8, s16, s32

