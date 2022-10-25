#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import copy
import os
import multiprocessing
import torch
import pandas as pd
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing import errors
from mup.coord_check import plot_coord_data
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


def cov(x):
    '''Treat `x` as a collection of vectors and its Gram matrix.
    Input:
        x: If it has shape [..., d], then it's treated as
            a collection of d-dimensional vectors
    Output:
        cov: a matrix of size N x N where N is the product of
            the non-last dimensions of `x`.
    '''
    if x.nelement() == 1:
        width = 1
        xx = x.reshape(1, 1)
    else:
        width = x.shape[-1]
        xx = x.reshape(-1, x.shape[-1])
    return xx @ xx.T / width


def covoffdiag(x):
    '''Get off-diagonal entries of `cov(x)` in a vector.
    Input:
        x: If it has shape [..., d], then it's treated as
            a collection of d-dimensional vectors
    Output:
        Off-diagonal entries of `cov(x)` in a vector.'''
    c = cov(x)
    return c[~torch.eye(c.shape[0], dtype=bool)]


FDICT = {
    'l1': lambda x: torch.abs(x).mean(),
    'l2': lambda x: (x**2).mean()**0.5,
    'mean': lambda x: x.mean(),
    'std': lambda x: x.std(),
    'covl1': lambda x: torch.abs(cov(x)).mean(),
    'covl2': lambda x: (cov(x)**2).mean()**0.5,
    'covoffdiagl1': lambda x: torch.abs(covoffdiag(x)).mean(),
    'covoffdiagl2': lambda x: (covoffdiag(x)**2).mean()**0.5
}


def convert_fdict(d):
    '''convert a dict `d` with string values to function values.
    Input:
        d: a dict whose values are either strings or functions
    Output:
        a new dict, with the same keys as `d`, but the string values are
        converted to functions using `FDICT`.
    '''
    return dict([
        ((k, FDICT[v]) if isinstance(v, str) else (k, v))
        for k, v in d.items()])


def _record_coords(records, width, modulename, t,
                   output_fdict=None, input_fdict=None, param_fdict=None):
    '''Returns a forward hook that records coordinate statistics.

    Returns a forward hook that records statistics regarding the output, input,
    and/or parameters of a `nn.Module`. This hook is intended to run only once,
    on the timestep specified by `t`.

    On forward pass, the returned hook calculates statistics specified in
    `output_fdict`, `input_fdict`, and `param_fdict`, such as the normalized l1
    norm, of output, input, and/or parameters of the module. The statistics are
    recorded along with the `width`, `modulename`, and `t` (the time step) as a
    dict and inserted into `records` (which should be a list). More precisely,
    for each output, input, and/or parameter, the inserted dict is of the form

        {
            'width': width, 'module': modified_modulename, 't': t,
            # keys are keys in fdict
            'l1': 0.241, 'l2': 0.420, 'mean': 0.0, ...
        }

    where `modified_modulename` is a string that combines the `modulename` with
    an indicator of which output, input, or parameter tensor is the statistics
    computed over.

    The `*_fdict` inputs should be dictionaries with string keys and whose
    values can either be functions or strings. The string values are converted
    to functions via `convert_fdict`. The default values of `*_dict` inputs are
    converted to `output_fdict = dict(l1=FDICT['l1'])`, `input_fdict = {}`,
    `param_fdict = {}`, i.e., only the average coordinate size (`l1`) of the
    output activations are recorded.

    Inputs:
        records:
            list to append coordinate data to
        width:
            width of the model. This is used only for plotting coord check later
            on, so it can be any notion of width.
        modulename:
            string name of the module. This is used only for plotting coord check.
        t:
            timestep of training. This is used only for plotting coord check.
        output_fdict, input_fdict, param_fdict:
            dicts with string keys and whose values can either be functions or
            strings. The string values are converted to functions via
            `convert_fdict`
    Output:
        a forward hook that records statistics regarding the output, input,
        and/or parameters of a `nn.Module`, as discussed above.
    '''
    if output_fdict is None:
        output_fdict = dict(l1=FDICT['l1'])
    else:
        output_fdict = convert_fdict(output_fdict)
    if input_fdict is None:
        input_fdict = {}
    else:
        input_fdict = convert_fdict(input_fdict)
    if param_fdict is None:
        param_fdict = {}
    else:
        param_fdict = convert_fdict(param_fdict)

    def f(module, input, output):
        def get_stat(d, x, fdict):
            if isinstance(x, (tuple, list)):
                for i, _x in enumerate(x):
                    _d = copy.copy(d)
                    _d['module'] += f'[{i}]'
                    get_stat(_d, _x, fdict)
            elif isinstance(x, dict):
                for name, _x in x.items():
                    _d = copy.copy(d)
                    _d['module'] += f'[{name}]'
                    get_stat(_d, _x, fdict)
            elif isinstance(x, torch.Tensor):
                _d = copy.copy(d)
                for fname, f in fdict.items():
                    _d[fname] = f(x).item()
                records.append(_d)
            else:
                raise NotImplemented(f'Unexpected output type: {type(x)}')

        with torch.no_grad():
            ret = {
                'width': width,
                'module': modulename,
                't': t
            }

            # output stats
            if isinstance(output, (tuple, list)):
                for i, out in enumerate(output):
                    _ret = copy.copy(ret)
                    _ret['module'] += f':out[{i}]'
                    get_stat(_ret, out, output_fdict)
            elif isinstance(output, dict):
                for name, out in output.items():
                    _ret = copy.copy(ret)
                    _ret['module'] += f':out[{name}]'
                    get_stat(_ret, out, output_fdict)
            elif isinstance(output, torch.Tensor):
                _ret = copy.copy(ret)
                for fname, f in output_fdict.items():
                    _ret[fname] = f(output).item()
                records.append(_ret)
            else:
                raise NotImplemented(f'Unexpected output type: {type(output)}')

            # input stats
            if input_fdict:
                if isinstance(input, (tuple, list)):
                    for i, out in enumerate(input):
                        _ret = copy.copy(ret)
                        _ret['module'] += f':in[{i}]'
                        get_stat(_ret, out, input_fdict)
                elif isinstance(input, dict):
                    for name, out in input.items():
                        _ret = copy.copy(ret)
                        _ret['module'] += f':in[{name}]'
                        get_stat(_ret, out, input_fdict)
                elif isinstance(input, torch.Tensor):
                    _ret = copy.copy(ret)
                    for fname, f in input_fdict.items():
                        _ret[fname] = f(input).item()
                    records.append(_ret)
                else:
                    raise NotImplemented(f'Unexpected output type: {type(input)}')

            # param stats
            if param_fdict:
                for name, p in module.named_parameters():
                    _ret = copy.copy(ret)
                    _ret['module'] += f':param[{name}]'
                    for fname, f in param_fdict.items():
                        _ret[fname] = f(p).item()
                    records.append(_ret)

    return f


def _get_coord_data(models, dataloader, optcls, nsteps=3,
                    dict_in_out=False, flatten_input=False, flatten_output=False,
                    output_name='loss', lossfn='xent', filter_module_by_name=None,
                    fix_data=True, cuda=True, nseeds=1,
                    output_fdict=None, input_fdict=None, param_fdict=None,
                    show_progress=True, one_hot_target=False):
    '''Inner method for `get_coord_data`.

    Train the models in `models` with optimizer given by `optcls` and data from
    `dataloader` for `nsteps` steps, and record coordinate statistics specified
    by `output_fdict`, `input_fdict`, `param_fdict`. By default, only `l1` is
    computed for output activations of each module.

    Inputs:
        models:
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optcls:
            a function so that `optcls(model)` gives an optimizer used to train
            the model.
        nsteps:
            number of steps to train the model
        dict_in_out:
            whether the data loader contains Huggingface-style dict input and
            output. Default: False
        flatten_input:
            if not `dict_in_out`, reshape the input to be
            `input.view(input.shape[0], -1)`. Typically used for testing MLPs.
        flatten_output:
            if not `dict_in_out`, reshape the label to be `label.view(-1,
            input.shape[-1])`.
        output_name:
            if `dict_in_out`, this is the key for the loss value if the output
            is a dict. If the output is not a dict, then we assume the first
            element of the output is the loss.
        lossfn:
            loss function to use if not `dict_in_out`. Can be either a string from
            [`xent`, 'mse', 'nll', 'l1'] or a python `callable` such that
            `lossfn(output, target)` returns the loss value. Examples of valid
            `callable`s are `F.cross_entropy`, `F.mse_loss`, etc, where `F` is
            `torch.nn.functional`. Default: 'xent'
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict:
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm. Default: True
        one_hot_target:
            convert target label into a one-hot vector. This typically is only
            used for `'mse'` or `'l1'` losses in classification tasks.
            Default: False
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).

    Breaking Changes:
        In v1.0.0, when `lossfn=='mse'`, the target is automatically converted
        to a one hot vector before loss computation. Starting in v1.1.0, this
        behavior is turned off, and the user needs to explicitly turn on this
        behavior by setting `one_hot_target=True`.

    '''
    df = []
    if fix_data:
        batch = next(iter(dataloader))
        dataloader = [batch] * nsteps
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=nseeds * len(models))

    for i in range(nseeds):
        torch.manual_seed(i)
        for width, model in models.items():
            model = model()
            model = model.train()
            if cuda:
                model = model.cuda()
            optimizer = optcls(model)
            for batch_idx, batch in enumerate(dataloader, 1):
                remove_hooks = []
                # add hooks
                for name, module in model.named_modules():
                    if filter_module_by_name and not filter_module_by_name(name):
                        continue
                    remove_hooks.append(module.register_forward_hook(
                        _record_coords(df, width, name, batch_idx,
                                       output_fdict=output_fdict,
                                       input_fdict=input_fdict,
                                       param_fdict=param_fdict)))
                data, target = batch["image"], batch["label"]
                if cuda:
                    data, target = data.cuda(), target.cuda()
                if flatten_input:
                    data = data.view(data.size(0), -1)
                output = model(data)
                if flatten_output:
                    output = output.view(-1, output.shape[-1])
                if one_hot_target:
                    target = F.one_hot(target,
                                       num_classes=output.size(-1)).float()
                if lossfn == 'xent':
                    loss = F.cross_entropy(output, target)
                elif lossfn == 'mse':
                    loss = F.mse_loss(output, target)
                elif lossfn == 'nll':
                    loss = F.nll_loss(output, target)
                elif lossfn == 'l1':
                    loss = F.l1_loss(output, target)
                elif callable(lossfn):
                    loss = lossfn(output, target)
                else:
                    raise NotImplementedError(f'unknown `lossfn`: {lossfn}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # remove hooks
                for handle in remove_hooks:
                    handle.remove()

                if batch_idx == nsteps: break
            if show_progress:
                pbar.update(1)
    if show_progress:
        pbar.close()
    return pd.DataFrame(df)


def get_coord_data(models, dataloader, optimizer='sgd', lr=None, mup=True,
                   filter_trainable_by_name=None,
                   **kwargs):
    '''Get coord data for coord check.

    Train the models in `models` with data from `dataloader` and optimizer
    specified by `optimizer` and `lr` for `nsteps` steps, and record coordinate
    statistics specified by `output_fdict`, `input_fdict`, `param_fdict`. By
    default, only `l1` is computed for output activations of each module.

    This function wraps around `_get_coord_data`, with the main difference being
    user can specify common optimizers via a more convenient interface.

    Inputs:
        models:
            a dict of lazy models, where the keys are numbers indicating width.
            Each entry of `models` is a function that instantiates a model given
            nothing.
        dataloader:
            an iterator whose elements are either Huggingface style dicts, if
            `dict_in_out` is True, or (input, label). If `fix_data` is True
            (which is the default), then only the first element of `dataloader`
            is used in a loop and the rest of `dataloder` is ignored.
        optimizer:
            a string in `['sgd', 'adam', 'adamw']`, with default being `'sgd'`.
        lr:
            learning rate. By default is 0.1 for `'sgd'` and 1e-3 for others.
        mup:
            If True, then use the optimizer from `mup.optim`; otherwise, use the
            one from `torch.optim`.
        filter_trainable_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be trained.
        nsteps:
            number of steps to train the model
        dict_in_out:
            whether the data loader contains Huggingface-style dict input and
            output. Default: False
        flatten_input:
            if not `dict_in_out`, reshape the input to be
            `input.view(input.shape[0], -1)`. Typically used for testing MLPs.
        flatten_output:
            if not `dict_in_out`, reshape the label to be `label.view(-1,
            input.shape[-1])`.
        output_name:
            if `dict_in_out`, this is the key for the loss value if the output
            is a dict. If the output is not a dict, then we assume the first
            element of the output is the loss.
        lossfn:
            loss function to use if not `dict_in_out`. Can be either a string from
            [`xent`, 'mse', 'nll', 'l1'] or a python `callable` such that
            `lossfn(output, target)` returns the loss value. Examples of valid
            `callable`s are `F.cross_entropy`, `F.mse_loss`, etc, where `F` is
            `torch.nn.functional`. Default: 'xent'
        filter_module_by_name:
            a function that returns a bool given module names (from
            `model.named_modules()`), or None. If not None, then only modules
            whose name yields True will be recorded.
        cuda:
            whether to use cuda or not. Default: True
        nseeds:
            number of times to repeat the training, each with different seeds.
        output_fdict, input_fdict, param_fdict:
            function dicts to be used in `_record_coords`. By default, only `l1`
            is computed for output activations of each module.
        show_progress:
            show progress using tqdm. Default: True
        one_hot_target:
            convert target label into a one-hot vector. This typically is only
            used for `'mse'` or `'l1'` losses in classification tasks.
            Default: False
    Output:
        a pandas DataFrame containing recorded results. The column names are
        `'width', 'module', 't'` as well as names of statistics recorded, such
        as `'l1'` (see `FDICT` for other premade statistics that can be
        collected).

    Breaking Changes:
        In v1.0.0, when `lossfn=='mse'`, the target is automatically converted
        to a one hot vector before loss computation. Starting in v1.1.0, this
        behavior is turned off, and the user needs to explicitly turn on this
        behavior by setting `one_hot_target=True`.
    '''
    if lr is None:
        lr = 0.1 if optimizer == 'sgd' else 1e-3
    if mup:
        from mup.optim import MuAdam as Adam
        from mup.optim import MuAdamW as AdamW
        from mup.optim import MuSGD as SGD
    else:
        from torch.optim import SGD, Adam, AdamW

    def get_trainable(model):
        params = model.parameters()
        if filter_trainable_by_name is not None:
            params = []
            for name, p in model.named_parameters():
                if filter_trainable_by_name(name):
                    params.append(p)
        return params

    if optimizer == 'sgd':
        optcls = lambda model: SGD(get_trainable(model), lr=lr)
    elif optimizer == 'adam':
        optcls = lambda model: Adam(get_trainable(model), lr=lr)
    elif optimizer == 'adamw':
        optcls = lambda model: AdamW(get_trainable(model), lr=lr)
    elif optimizer is None:
        raise ValueError('optimizer should be sgd|adam|adamw or a custom function')

    data = _get_coord_data(models, dataloader, optcls, **kwargs)
    data['optimizer'] = optimizer
    data['lr'] = lr
    return data


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
