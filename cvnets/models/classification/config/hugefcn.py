#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Dict


def get_configuration(opts) -> Dict:

    width_multiplier = getattr(opts, "model.classification.hugefcn.width_multiplier", 1.0)
    patch_size = getattr(opts, "model.classification.hugefcn.patch_size", 16)

    dim = int(1024 * width_multiplier)

    config = {
        "general": {
            "img_channels": 3,
            "dim": dim,
            "patch_size": patch_size

        },
    }
    return config
