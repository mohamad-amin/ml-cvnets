'''
Optimizers with μP scaling.

Here we provide 3 ready-to-go optimizers MuAdam, MuAdamW, and MuSGD.
However, the user can easily convert their own optimizer to a μP
optimizer: if your `optimizer` is "Adam-like", such as RMSProp and Adagrad,
that involves normalizing the gradient entrywise, then the following creates
the desired μP optimizer:

    def MuOptimizer(params, **kwargs):
        return MuAdam(params, impl=optimizer, **kwargs)

On the other hand, if your `optimizer` is "SGD-like", such as ASGD, then
the following creates the desired μP optimizer:

    def MuOptimizer(params, **kwargs):
        return MuSGD(params, impl=optimizer, **kwargs)

See Appendix B in our paper for discussions of other optimizers.
'''
from collections import defaultdict


def get_muadam_param_groups(model_params):

    new_param_groups = []
    lr_mults = []
    for param_group in model_params:
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != 'params'}
            new_g['params'] = []
            return new_g

        matrix_like_p = defaultdict(new_group)  # key is width_mult
        vector_like_p = new_group()
        for p in param_group['params']:
            assert hasattr(p, 'infshape'), (
                f'A parameter with shape {p.shape} does not have `infshape` attribute. '
                'Did you forget to call `mup.set_base_shapes` on the model?')
            if p.infshape.ninf() == 2:
                matrix_like_p[p.infshape.width_mult()]['params'].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError('more than 2 inf dimensions')
            else:
                vector_like_p['params'].append(p)

        matrix_like_lr_mults = []
        for width_mult, group in matrix_like_p.items():
            # Scale learning rate and weight decay accordingly
            matrix_like_lr_mults.append(1. / width_mult)
            group['weight_decay'] *= width_mult
        new_param_groups.extend(list(matrix_like_p.values()) + [vector_like_p])

        lr_mults.extend(matrix_like_lr_mults + [1.0] * len([vector_like_p]))

    return new_param_groups, lr_mults


def get_musgd_param_groups(model_params):

    new_param_groups = []
    lr_mults = []
    for param_group in model_params:
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != 'params'}
            new_g['params'] = []
            return new_g

        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        vector_like_p = defaultdict(new_group)  # key is width mult
        matrix_like_p = defaultdict(new_group)  # key is fan_in/out ratio
        fixed_p = new_group()

        for p in param_group['params']:
            assert hasattr(p, 'infshape'), (
                f'A parameter with shape {p.shape} does not have `infshape` attribute. '
                'Did you forget to call `mup.set_base_shapes` on the model?')
            if p.infshape.ninf() == 1:
                vector_like_p[p.infshape.width_mult()]['params'].append(p)
            elif p.infshape.ninf() == 2:
                matrix_like_p[p.infshape.fanin_fanout_mult_ratio()]['params'].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError('more than 2 inf dimensions')
            else:
                fixed_p['params'].append(p)

        vector_like_lr_mults = []
        for width_mult, group in vector_like_p.items():
            # Scale learning rate and weight decay accordingly
            vector_like_lr_mults.append(width_mult)
            group['weight_decay'] /= width_mult
        matrix_like_lr_mults = []
        for shape_ratio, group in matrix_like_p.items():
            matrix_like_lr_mults.append(1. / shape_ratio)
            group['weight_decay'] *= shape_ratio

        new_param_groups.extend(
            list(matrix_like_p.values()) +
            list(vector_like_p.values()) +
            [fixed_p]
        )
        lr_mults.extend(
            matrix_like_lr_mults +
            vector_like_lr_mults +
            [1.0] * len([fixed_p])
        )

    return new_param_groups, lr_mults
