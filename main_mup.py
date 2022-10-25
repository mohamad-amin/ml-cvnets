#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import copy
import os
import multiprocessing
import torch
from torch.cuda.amp import GradScaler
from torch.distributed.elastic.multiprocessing import errors
from mup.coord_check import get_coord_data, plot_coord_data
from mup import get_shapes, make_base_shapes, set_base_shapes

from utils import logger
from options.opts import get_training_arguments
from utils.common_utils import device_setup, create_directories
from utils.ddp_utils import is_master, distributed_init
from cvnets import get_model
from loss_fn import build_loss_fn
from optim import build_optimizer
from optim.scheduler import build_scheduler
from data import create_train_val_loader
from common import (
    DEFAULT_EPOCHS,
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_EPOCHS,
)


def coord_check(mup, trainloader, optimizer, get_model, base_shapes, plotdir='', legend=False):

    optimizer = optimizer.replace('mu', '')

    def gen(w, standparam=False):
        def f():
            model = get_model(w)
            if standparam:
                set_base_shapes(model, None)
            else:
                set_base_shapes(model, base_shapes)
            return model
        return f

    widths = 2 ** torch.arange(-2., 2)
    models = {w: gen(w, standparam=not mup) for w in widths}
    df = get_coord_data(models, trainloader, mup=mup, lr=0.1, optimizer=optimizer, nseeds=3, nsteps=5, dict_in_out=True)

    prm = 'μP' if mup else 'SP'
    plot_coord_data(df, legend=legend,
        save_to=os.path.join(plotdir, f'{prm.lower()}_{optimizer}_coord.png'),
        suptitle=f'{prm} {optimizer} lr={0.1} nseeds={5}',
        face_color='xkcd:light grey' if not mup else None)


@errors.record
def main(opts, **kwargs):

    num_gpus = getattr(opts, "dev.num_gpus", 0)  # defaults are for CPU
    dev_id = getattr(opts, "dev.device_id", torch.device("cpu"))
    device = getattr(opts, "dev.device", torch.device("cpu"))

    is_master_node = is_master(opts)

    # set-up data loaders
    train_loader, val_loader, train_sampler = create_train_val_loader(opts)

    # compute max iterations based on max epochs
    # Useful in doing polynomial decay
    is_iteration_based = getattr(opts, "scheduler.is_iteration_based", False)
    if is_iteration_based:
        max_iter = getattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
        if max_iter is None or max_iter <= 0:
            logger.log("Setting max. iterations to {}".format(DEFAULT_ITERATIONS))
            setattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
            max_iter = DEFAULT_ITERATIONS
        setattr(opts, "scheduler.max_epochs", DEFAULT_MAX_EPOCHS)
        if is_master_node:
            logger.log("Max. iteration for training: {}".format(max_iter))
    else:
        max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        if max_epochs is None or max_epochs <= 0:
            logger.log("Setting max. epochs to {}".format(DEFAULT_EPOCHS))
            setattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        setattr(opts, "scheduler.max_iterations", DEFAULT_MAX_ITERATIONS)
        max_epochs = getattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
        if is_master_node:
            logger.log("Max. epochs for training: {}".format(max_epochs))

    # memory format
    memory_format = (
        torch.channels_last
        if getattr(opts, "common.channels_last", False)
        else torch.contiguous_format
    )

    optim_name = getattr(opts, "optim.name", "sgd").lower()

    width_multiplier_key = getattr(opts, "mup.width_multiplier_key")

    def get_model_by_width(w):
        new_opts = copy.deepcopy(opts)
        setattr(new_opts, width_multiplier_key, w)
        new_model = get_model(new_opts)
        new_model = new_model.to(device=device, memory_format=memory_format)
        return new_model

    save_base_shapes = getattr(opts, "mup.save_base_shapes", False)
    if save_base_shapes:
        default_wm = getattr(opts, width_multiplier_key, 1.0)
        base_shapes = get_shapes(get_model(opts))
        delta_shapes = get_shapes(get_model_by_width(default_wm / 2))
        make_base_shapes(base_shapes, delta_shapes, savefile=save_base_shapes)
        logger.log("Saved base shapes in {}".format(save_base_shapes))
        exit()

    base_shapes = getattr(opts, "mup.load_base_shapes", False)
    if not base_shapes:
        logger.error("mup.load_base_shapes not defined! Exiting.")
        exit()

    logger.log("Testing parametrization")
    plot_dir = getattr(opts, "mup.coord_check_dir", "/tmp/coord_checks")
    import os
    os.makedirs(plot_dir, exist_ok=True)
    coord_check(mup=True,
                trainloader=train_loader, optimizer=optim_name,
                get_model=get_model_by_width, base_shapes=base_shapes, plotdir=plot_dir, legend=False)
    coord_check(mup=False,
                trainloader=train_loader, optimizer=optim_name,
                get_model=get_model_by_width, base_shapes=base_shapes, plotdir=plot_dir, legend=False)


def main_worker(**kwargs):
    opts = get_training_arguments()
    print(opts)
    # device set-up
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank", 0)
    if node_rank < 0:
        logger.error("--rank should be >=0. Got {}".format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc", "results")
    run_label = getattr(opts, "common.run_label", "run_1")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    num_gpus = getattr(opts, "dev.num_gpus", 1)
    setattr(opts, "ddp.use_distributed", False)

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = multiprocessing.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)
    norm_name = getattr(opts, "model.normalization.name", "batch_norm")

    if dataset_workers == -1:
        setattr(opts, "dataset.workers", n_cpus)

    if norm_name in ["sync_batch_norm", "sbn"]:
        setattr(opts, "model.normalization.name", "batch_norm")

    # adjust the batch size
    train_bsize = getattr(opts, "dataset.train_batch_size0", 32) * max(1, num_gpus)
    val_bsize = getattr(opts, "dataset.val_batch_size0", 32) * max(1, num_gpus)
    setattr(opts, "dataset.train_batch_size0", train_bsize)
    setattr(opts, "dataset.val_batch_size0", val_bsize)
    setattr(opts, "dev.device_id", None)
    main(opts=opts, **kwargs)


if __name__ == "__main__":
    #
    main_worker()
    # Todo:
    # Load base shapes in main train
