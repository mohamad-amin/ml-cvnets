#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn
import argparse

from . import register_cls_models
from .base_cls import BaseEncoder
from .config.hugefcn import get_configuration
from ...layers import ConvLayer, LinearLayer, GlobalPool, Identity


@register_cls_models("hugefcn")
class HugeFCN(BaseEncoder):
    """
    This class defines the HugeFCN architecture
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        hugefcn_config = get_configuration(opts=opts)
        image_channels = hugefcn_config["general"]["img_channels"]
        dim = hugefcn_config["general"]["dim"]

        super().__init__(*args, **kwargs)

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            opts=opts,
            in_channels=image_channels,
            out_channels=dim,
            kernel_size=224,  # Todo: LMAO
            stride=1,
            use_norm=True,
            use_act=True,
        )
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": dim}

        self.layer_1 = GlobalPool(pool_type=pool_type, keep_dim=False)
        self.layer_2 = nn.Sequential(LinearLayer(dim, dim), nn.ReLU())
        self.layer_3 = nn.Sequential(LinearLayer(dim, dim), nn.ReLU())
        self.layer_4 = nn.Sequential(LinearLayer(dim, dim), nn.ReLU())
        self.layer_5 = nn.Sequential(LinearLayer(dim, dim), nn.ReLU())

        self.conv_1x1_exp = Identity()

        self.classifier = LinearLayer(in_features=dim, out_features=num_classes)

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--model.classification.hugefcn.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier. Defaults to 1.0",
        )
        return parser