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
import os
import time
from model.decode import Decode

if __name__ == '__main__':
    file = 'data/coco_classes.txt'

    # model_path = 'pretrained_resnet50'
    # model_path = 'pretrained_resnet101'
    # model_path = 'pretrained_resnet50dcn'
    # model_path = 'pretrained_darknet53'
    model_path = './weights/step1066300-ep073-loss5.715.pd'

    backbone_names = ['resnet50', 'resnet101', 'resnet50dcn', 'darknet53']
    backbone_name = backbone_names[2
    ]

    use_gpu = True
    import platform
    sysstr = platform.system()
    if sysstr == 'Windows':
        use_gpu = False
        use_gpu = True

    _decode = Decode(backbone_name, 550, 0.05, 0.5, model_path, file, use_gpu=use_gpu)

    kk = 0
    # warm up
    for (root, dirs, files) in os.walk('images/test'):
        if files:
            for f in files:
                path = os.path.join(root, f)
                image = cv2.imread(path)
                image, boxes, masks, classes, scores = _decode.detect_image(image, top_k=100, draw=False)
                kk += 1
                print(kk)
                if kk == 10:
                    break

    for (root, dirs, files) in os.walk('images/test'):
        if files:
            start = time.time()
            for f in files:
                path = os.path.join(root, f)
                image = cv2.imread(path)
                print(path)
                start2 = time.time()
                image, boxes, masks, classes, scores = _decode.detect_image(image, top_k=100, draw=False)
                # image, boxes, masks, classes, scores = _decode.detect_image(image, top_k=100, draw=True)
                print('time: {0:.6f}s'.format(time.time() - start2))
                cv2.imwrite('images/res/' + f, image)
            cost = time.time() - start
            num_imgs = 98
            print('total time: {0:.6f}s'.format(cost))
            print('Speed: %.6fs per image,  %.1f FPS.' % ((cost / num_imgs), (num_imgs / cost)))


