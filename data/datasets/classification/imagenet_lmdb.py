#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torchvision.datasets import ImageFolder
from typing import Optional, Tuple, Dict, List, Union
import os
import torch
import lmdb
import six
import pyarrow as pa
from PIL import Image
import argparse

from utils import logger

from .. import register_dataset
from ..dataset_base import BaseImageDataset
from ...transforms import image_pil as T
from ...collate_fns import register_collate_fn


@register_dataset(name="imagenet_lmdb", task="classification")
class ImagenetLMDBDataset(BaseImageDataset):

    def __init__(
        self,
        opts,
        is_training: Optional[bool] = True,
        is_evaluation: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:

        if getattr(opts, "dataset.trove.enable", False):
            raise NotImplementedError()

        self.root = (
            getattr(opts, "dataset.root_train", None)
            if is_training
            else getattr(opts, "dataset.root_val", None)
        )
        root = self.root

        self.is_training = is_training
        self.is_evaluation = is_evaluation
        self.sampler_name = getattr(opts, "sampler.name", None)
        self.opts = opts

        image_device_cuda = getattr(self.opts, "dataset.decode_data_on_gpu", False)
        if image_device_cuda:
            raise NotImplementedError()

        self.device = getattr(self.opts, "dev.device", torch.device("cpu"))

        if getattr(opts, "dataset.cache_images_on_ram", False):
            raise NotImplementedError()

        self.n_classes = len(list(self.class_to_idx.keys()))
        setattr(opts, "model.classification.n_classes", self.n_classes)
        setattr(opts, "dataset.collate_fn_name_train", "imagenet_collate_fn")
        setattr(opts, "dataset.collate_fn_name_val", "imagenet_collate_fn")
        setattr(opts, "dataset.collate_fn_name_eval", "imagenet_collate_fn")

        self.env = lmdb.open(root, subdir=os.path.isdir(root),
                             readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--dataset.imagenet.crop-ratio",
            type=float,
            default=0.875,
            help="Crop ratio",
        )
        return parser

    def _training_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
            Training data augmentation methods.
                Image --> RandomResizedCrop --> RandomHorizontalFlip --> Optional(AutoAugment or RandAugment)
                --> Tensor --> Optional(RandomErasing) --> Optional(MixUp) --> Optional(CutMix)

        .. note::
            1. AutoAugment and RandAugment are mutually exclusive.
            2. Mixup and CutMix are applied on batches are implemented in trainer.
        """
        aug_list = [
            T.RandomResizedCrop(opts=self.opts, size=size),
            T.RandomHorizontalFlip(opts=self.opts),
        ]
        auto_augment = getattr(
            self.opts, "image_augmentation.auto_augment.enable", False
        )
        rand_augment = getattr(
            self.opts, "image_augmentation.rand_augment.enable", False
        )
        if auto_augment and rand_augment:
            logger.error(
                "AutoAugment and RandAugment are mutually exclusive. Use either of them, but not both"
            )
        elif auto_augment:
            aug_list.append(T.AutoAugment(opts=self.opts))
        elif rand_augment:
            aug_list.append(T.RandAugment(opts=self.opts))

        aug_list.append(T.ToTensor(opts=self.opts))

        if getattr(self.opts, "image_augmentation.random_erase.enable", False):
            aug_list.append(T.RandomErasing(opts=self.opts))

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _validation_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """
        Validation augmentation
            Image --> Resize --> CenterCrop --> ToTensor
        """
        aug_list = [
            T.Resize(opts=self.opts),
            T.CenterCrop(opts=self.opts),
            T.ToTensor(opts=self.opts),
        ]

        return T.Compose(opts=self.opts, img_transforms=aug_list)

    def _evaluation_transforms(self, size: Union[Tuple, int], *args, **kwargs):
        """Same as the validation_transforms"""
        return self._validation_transforms(size=size)

    def read_image_lmdb(self, index):

        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        return img, target

    def __getitem__(self, batch_indexes_tup: Tuple) -> Dict:
        """
        :param batch_indexes_tup: Tuple of the form (Crop_size_W, Crop_size_H, Image_ID)
        :return: dictionary containing input image, label, and sample_id.
        """
        crop_size_h, crop_size_w, img_index = batch_indexes_tup
        if self.is_training:
            transform_fn = self._training_transforms(size=(crop_size_h, crop_size_w))
        else:
            # same for validation and evaluation
            transform_fn = self._validation_transforms(size=(crop_size_h, crop_size_w))

        input_img, target = self.read_image_lmdb(img_index)

        if input_img is None:
            # Sometimes images are corrupt
            # Skip such images
            logger.log("Img index {} is possibly corrupt.".format(img_index))
            input_tensor = torch.zeros(
                size=(3, crop_size_h, crop_size_w), dtype=self.img_dtype
            )
            target = -1
            data = {"image": input_tensor}
        else:
            data = {"image": input_img}
            data = transform_fn(data)

        data["label"] = target
        data["sample_id"] = img_index

        return data

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        from utils.tensor_utils import image_size_from_opts

        im_h, im_w = image_size_from_opts(opts=self.opts)

        if self.is_training:
            transforms_str = self._training_transforms(size=(im_h, im_w))
        else:
            transforms_str = self._validation_transforms(size=(im_h, im_w))

        return "{}(\n\troot={}\n\tis_training={}\n\tsamples={}\n\tn_classes={}\n\ttransforms={}\n)".format(
            self.__class__.__name__,
            self.root,
            self.is_training,
            self.length,
            self.n_classes,
            transforms_str,
        )