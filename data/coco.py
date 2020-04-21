import os
import os.path as osp
import sys
import cv2
import numpy as np
import random



COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}

def get_label_map():
    return COCO_LABEL_MAP

class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                if label_idx >= 0:
                    label_idx = self.label_map[label_idx] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res


class COCODetection(object):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, transform=None,
                 dataset_name='MS COCO', has_gt=True):
        # Do this here because we have too many things named COCO
        from pycocotools.coco import COCO

        self.root = image_path
        self.coco = COCO(info_file)
        
        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())
        
        self.transform = transform
        self.target_transform = COCOAnnotationTransform()
        
        self.name = dataset_name
        self.has_gt = has_gt

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]

        if self.has_gt:
            # 找出这张图片的所有注解，一个注解代表一个物体。返回list
            target = self.coco.imgToAnns[img_id]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)    # 返回这张图片的所有注解的id，一个注解代表一个物体。返回list

            # Target has {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)     # 重新赋值一次target，其实target的值没有变。
        else:
            target = []

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.

        # 将上一步所得的target分为两部分。crowd是一群物体，target是单个物体。
        crowd  = [x for x in target if     ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)   # 这张图片 一群物体 的注解数目

        for x in crowd:   # 如果注解 是标注了一群物体，将其类别id设置为-1
            x['category_id'] = -1

        # This is so we ensure that all crowd annotations are at the end of the array
        target += crowd   # 再次合并 两部分 为target。在target数组中，我们保证了群体标记放在了最后。
        
        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.root, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)
        
        img = cv2.imread(path)    # 用你最熟悉的cv2打开图片。是bgr格式
        height, width, _ = img.shape
        
        if len(target) > 0:    # 如果有注解。注意，coco2017_val有48张图片没有注解，即图片上没有80种物体的任何一种。
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            # mask是一个list，里面每一个元素是一个shape=(height*width,)的ndarray，1代表是这个注解注明的物体，0代表其他物体和背景。
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)   # 设N=len(mask)=注解数，这一步将mask转换成一个ndarray，shape=(N, height*width)
            masks = masks.reshape(-1, height, width)   # shape=(N, height, width)

        # 读target中的N个注解的bbox，赋值给target
        # target = [[x0, y0, x1, y1, class_id], ...]  是一个二维list。左上角、右下角坐标做了归一化（除以图片宽高）。
        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)

        # 数据增强，训练集用的是SSDAugmentation(MEANS)，验证集用BaseTransform(MEANS)
        if len(target) > 0:
            target = np.array(target)
            # 默认resnet。减均值除以标准差。返回HWC且是RGB格式的图片。
            img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                {'num_crowds': num_crowds, 'labels': target[:, 4]})

            # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
            num_crowds = labels['num_crowds']
            labels     = labels['labels']
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        if target.shape[0] == 0:    # 如果没有注解。数据增强后可能没有。注意，coco2017_val有48张图片没有注解，即图片上没有80种物体的任何一种。
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids)-1))  # 如果没有注解。丢弃这张，随机选另外一张图片。

        # 图片ndarray转Tensor，换位CHW格式。target是这一张图片的bboxes注解。
        # target(13, 5)   mask(13, 550, 550)   num_crowds: int32 = 0
        return img.transpose(2, 0, 1).astype(np.float32), target, masks, num_crowds

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

