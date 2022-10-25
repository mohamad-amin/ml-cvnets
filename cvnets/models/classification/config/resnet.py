#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Dict

from utils import logger


def get_configuration(opts) -> Dict:
    depth = getattr(opts, "model.classification.resnet.depth", 50)
    resnet_config = dict()

    base_width = 64
    growth_factor = getattr(opts, "model.classification.resnet.growth_factor", 2)
    width_multiplier = getattr(opts, "model.classification.resnet.width_multiplier", 1)

    widths = [int(base_width * width_multiplier * (growth_factor ** i)) for i in range(4)]

    if depth == 18:
        resnet_config["layer1"] = {
            "input_channels": widths[0]
        }
        resnet_config["layer2"] = {
            "num_blocks": 2,
            "mid_channels": widths[0],
            "block_type": "basic",
            "stride": 1,
        }
        resnet_config["layer3"] = {
            "num_blocks": 2,
            "mid_channels": widths[1],
            "block_type": "basic",
            "stride": 2,
        }
        resnet_config["layer4"] = {
            "num_blocks": 2,
            "mid_channels": widths[2],
            "block_type": "basic",
            "stride": 2,
        }
        resnet_config["layer5"] = {
            "num_blocks": 2,
            "mid_channels": widths[3],
            "block_type": "basic",
            "stride": 2,
        }
    elif depth == 34:
        resnet_config["layer1"] = {
            "input_channels": widths[0]
        }
        resnet_config["layer2"] = {
            "num_blocks": 3,
            "mid_channels": widths[0],
            "block_type": "basic",
            "stride": 1,
        }
        resnet_config["layer3"] = {
            "num_blocks": 4,
            "mid_channels": widths[1],
            "block_type": "basic",
            "stride": 2,
        }
        resnet_config["layer4"] = {
            "num_blocks": 6,
            "mid_channels": widths[2],
            "block_type": "basic",
            "stride": 2,
        }
        resnet_config["layer5"] = {
            "num_blocks": 3,
            "mid_channels": widths[3],
            "block_type": "basic",
            "stride": 2,
        }
    elif depth == 50:
        resnet_config["layer1"] = {
            "input_channels": widths[0]
        }
        resnet_config["layer2"] = {
            "num_blocks": 3,
            "mid_channels": widths[0],
            "block_type": "bottleneck",
            "stride": 1,
        }
        resnet_config["layer3"] = {
            "num_blocks": 4,
            "mid_channels": widths[1],
            "block_type": "bottleneck",
            "stride": 2,
        }
        resnet_config["layer4"] = {
            "num_blocks": 6,
            "mid_channels": widths[2],
            "block_type": "bottleneck",
            "stride": 2,
        }
        resnet_config["layer5"] = {
            "num_blocks": 3,
            "mid_channels": widths[3],
            "block_type": "bottleneck",
            "stride": 2,
        }
    elif depth == 101:
        resnet_config["layer1"] = {
            "input_channels": widths[0]
        }
        resnet_config["layer2"] = {
            "num_blocks": 3,
            "mid_channels": widths[0],
            "block_type": "bottleneck",
            "stride": 1,
        }
        resnet_config["layer3"] = {
            "num_blocks": 4,
            "mid_channels": widths[1],
            "block_type": "bottleneck",
            "stride": 2,
        }
        resnet_config["layer4"] = {
            "num_blocks": 23,
            "mid_channels": widths[2],
            "block_type": "bottleneck",
            "stride": 2,
        }
        resnet_config["layer5"] = {
            "num_blocks": 3,
            "mid_channels": widths[3],
            "block_type": "bottleneck",
            "stride": 2,
        }
    else:
        logger.error(
            "ResNet models are supported with depths of 18, 34, 50 and 101. Please specify depth using "
            "--model.classification.resnet.depth flag. Got: {}".format(depth)
        )
    return resnet_config
