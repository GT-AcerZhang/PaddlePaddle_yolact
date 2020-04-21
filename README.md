# YOLACT
其实并不是完美复刻，只能说大部分情况跟随了原仓库。

## 更新日记

2020/04/22:初次见面

## 需要补充

从ImageNet预训练的resnet50dcn开始训练

## 概述

mask AP为30.5。这是飞桨论文复现挑战赛参赛作品其一，并不是我提交的最终版本。

## 官方权重转换

1_pytorch2paddle.py将yolact作者的pytorch模型转换为paddlepaddle模型。（地址https://github.com/dbolya/yolact）
再对转换好的模型finetune，减少训练时间。
唯一import torch的地方。
这个脚本在本地windows上跑，再把pretrained_resnet50dcn模型上传到AIStudio。如果不允许，请从头训练。

## 训练
train.py用于训练
有两种训练模型。0-从头训练，1-读取模型继续训练。通过指定--pattern参数指定。
如果你要从头训练，键入下面命令：
```
nohup python train.py --initial_step=0 --conf_loss=ce_loss --pattern=0>> train.txt 2>&1 &

```

如果你从转换后的pretrained_resnet50dcn模型训练，键入下面命令：
```
nohup python train.py --initial_step=800000 --steps=830000 --conf_loss=ce_loss --pattern=1 --model_path=./pretrained_resnet50dcn>> train.txt 2>&1 &

```
我试过，训练30000步之后就能达到30% mAP以上。


如果想接着从上次训练后的模型（比如是step1066300-ep073-loss5.715.pd）继续训练，键入下面命令：
```
nohup python train.py --initial_step=1066300 --steps=1230000 --conf_loss=ce_loss --pattern=1 --model_path=./weights/step1066300-ep073-loss5.715.pd>> train.txt 2>&1 &

```

## 评估
test_dev.py用于跑COCO test-dev的图片，最后会生成一个json文件
results/detections_test-dev2017_yolactplus_results.json
提交到
https://competitions.codalab.org/competitions/20796#participate
进行评分。

eval.py用于计算val2017的mAP

## 预测
demo.py用于预测images/test/下的图片，结果保存在images/res/目录下。
一些参数（model_path、backbone_name、obj_threshold、nms_threshold等）在代码中改。

## 传送门
cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。


## 广告位招租
本人单身，有意向相处的理工科妹子请加微信wer186259（推荐同学、朋友亦可），备注GitHub，并带上一小段自我介绍。关于我本人，本科毕业于南京大学（2018），人在广州，爱好GitHub，爱好开源。希望对方是本科或者硕士学历，专业嘛。。。能看到这个仓库的也只能是那几种专业了，一起讨论算法和编程也是很有趣的！希望对方能粘人一点，主动一点，我周末也基本在研究算法，我一直也没有什么时间社交，很被动，这次是我这么认真地对待这件事。

# Citation

```
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```

```
@misc{yolact-plus-arxiv2019,
  title         = {YOLACT++: Better Real-time Instance Segmentation},
  author        = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  year          = {2019},
  eprint        = {1912.06218},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
