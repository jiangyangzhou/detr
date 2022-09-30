# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == "gaze_follow"  :
        transform = _get_transform()
        gazefollow_train_data = "/mnt/sdb1/gazeFollow/gazefollow_extend"
        gazefollow_train_label = "/mnt/sdb1/gazeFollow/gazefollow_extend/train_annotations_release.txt"
        gazefollow_val_data = "/mnt/sdb1/gazeFollow/gazefollow_extend"
        gazefollow_val_label = "/mnt/sdb1/gazeFollow/gazefollow_extend/test_annotations_release.txt"
        train_dataset = GazeFollow(gazefollow_train_data, gazefollow_train_label,
                      transform, input_size=input_resolution, output_size=output_resolution)
        return train_dataset
    raise ValueError(f'dataset {args.dataset_file} not supported')

