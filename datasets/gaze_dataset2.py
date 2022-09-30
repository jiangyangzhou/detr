import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import pandas as pd

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
# from scipy.misc import imresize
from skimage.transform import resize as imresize


import os
import glob
import csv

from .utils import imutils
from .utils import myutils
# from ..config import *
from detectron2.data.detection_utils import annotations_to_instances
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import datasets.transforms as T
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
input_resolution = 448
output_resolution = 64
class GazeFollow(Dataset):
    def __init__(self, data_dir, csv_path, transform, input_size=input_resolution, output_size=output_resolution,
                 test=False, imshow=False):
        if test:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
                    'bbox_y_max']].groupby(['path', 'eye_x'])
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)
        else:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df[df['inout'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            df.reset_index(inplace=True)
            self.y_train = df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'eye_x', 'eye_y', 'gaze_x',
                               'gaze_y', 'inout']]
            self.X_train = df['path']
            self.length = len(df)

        self.data_dir = data_dir
        self.transform = transform
        self.test = test

        self.input_size = input_size
        self.output_size = output_size
        self.imshow = imshow
        self.prepare = ConvertCocoPolysToMask(False)
    def __getitem__(self, index):
        if self.test:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            bbox = []
            annotations = []
            for i, row in g.iterrows():
                path = row['path']
                x_min = row['bbox_x_min']
                y_min = row['bbox_y_min']
                x_max = row['bbox_x_max']
                y_max = row['bbox_y_max']
                eye_x = row['eye_x']
                eye_y = row['eye_y']
                gaze_x = row['gaze_x']
                gaze_y = row['gaze_y']
                bbox.append( list(map(float, [x_min, y_min, x_max, y_max])) )
                cont_gaze.append([gaze_x, gaze_y])  # all ground truth gaze are stacked up
                obj = {}
                obj["bbox"] = [x_min, y_min, x_max, y_max]
                # obj["bbox_mode"] = BoxMode.XYWH_ABS
                obj["keypoints"] = [gaze_x, gaze_y]
                obj["image_id"] == index
                obj["category_id"] = 1
                obj["gaze_inside"] = torch.tensor([1]),
                annotations.append(obj)
            # for j in range(len(cont_gaze), 20):
            #     cont_gaze.append([-1, -1])  # pad dummy gaze to match size for batch processing
            #     bbox.append([-1,-1,-1,-1])
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = True # always consider test samples as inside
        else:
            path = self.X_train.iloc[index]
            x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, inout = self.y_train.iloc[index]
            gaze_inside = bool(inout)


        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        width, height = img.size
        imsize = torch.IntTensor([width, height])
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        if self.test:  # aggregated heatmap
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                         3,
                                                         type='Gaussian')
            gaze_heatmap /= num_valid
        else:
            # if gaze_inside:
            gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')
    
        
        if self.test:
            data = { #"image":img,
                # "face": face,
                # "head_channel": head_channel,
                # "gaze_heatmap": gaze_heatmap, 
                "cont_gaze": cont_gaze, 
                "imgsize": imsize,
                "path": path, 
                "annotations": annotations
            }   
            img, target = self.prepare(img, data)
            # data["instances"] = annotations_to_instances(data["annotation"], imsize)
            return  img, data
        else:
            data = { #"image":img,
                # "face": face,
                 "image_id": index,
                # "head_channel": head_channel,
                # "path": path,
                # "imgsize": imsize,
                "annotations": [{
                    "id": 1,    
                    "image_id": index,
                    "category_id": 1,
                    "bbox": list(map(float, [x_min, y_min, x_max, y_max])),
                    "keypoint": list(map(float, [gaze_x, gaze_y])),
                    "gaze_inside": torch.tensor([gaze_inside]),
                    "gaze_heatmap": gaze_heatmap, 
                    # "bbox_mode": BoxMode.XYWH_ABS
                    }
                ]
            }   

            img, target = self.prepare(img, data)
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            # data["instances"] = annotations_to_instances(data["annotation"], imsize)
            return img, target

    def __len__(self):
        return self.length


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # target["area"] = area[keep]
        # target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
