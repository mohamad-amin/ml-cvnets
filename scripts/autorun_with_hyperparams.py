import math

import yaml
import argparse
import subprocess
from pathlib import Path
from copy import deepcopy
from datetime import date

qsub_submit_script = '''#!/bin/bash

#PBS -l walltime={walltime}:00:00,select=1:ncpus={ncpus}:ngpus={ngpus}:gpu_mem={gpu_mem}:mem={mem}
#PBS -N {job_index}.{name}
#PBS -A {allocation_code}
#PBS -m abe
#PBS -M {user_mail}
#PBS -o {run_dir}/output_{job_index}.txt
#PBS -e {run_dir}/error_{job_index}.txt

################################################################################

source /home/lemohama/.bashrc
module restore 2021_gpu

{cuda_visible_devices_expr}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

conda activate old_jax
conda deactivate
conda activate old_jax

export JAX_ENABLE_X64=True

mkdir /tmp/imagenet/
cd /tmp/imagenet/
cp /scratch/st-dsuth-1/amin/datasets/imagenet_zips/* .
rm valprep.sh
./extract.bsh
rm -rf imagenet/train/ILSVRC2012_img_train
rm imagenet/val/ILSVRC2012_img_val.tar

cd $PBS_O_WORKDIR

python main_train.py --common.config-file {config_path} --common.results-loc {run_dir}/run_{job_index}/
'''

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def save_config(config_dict, config_path):
    with open(config_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=None)


def add_explanation(explanation, key, value):
    return explanation + ('' if explanation == '' else '-__-') + '{}={}'.format(key, value)


def make_and_get(parent, child):
    if child not in parent:
        parent[child] = {}
    return parent[child]


def update_attribute_in_config(base_config, attribute, new_value, inplace=False):
    hierarchy = attribute.split('_--_')
    config = deepcopy(base_config)
    attribute = config
    for item in hierarchy[:-1]:
        attribute = make_and_get(attribute, item)
    attribute[hierarchy[-1]] = new_value
    return config


def generate_configs(base_config, hyperparam_config):

    def extract_configs_from_attribute_tree(tree, key, base_config, explanation):
        for branch in tree.keys():
            yield from extract_configs_from_branch(tree[branch], key, base_config, explanation=explanation)

    def extract_configs_from_branch(tree, parent_key, base_config, explanation):
        for i, value in enumerate(tree['__value']):

            new_config = update_attribute_in_config(base_config, parent_key, value)
            new_explanation = add_explanation(explanation, parent_key, value)
            correspondings = tree.get('__correspondings', None)
            if correspondings is not None:
                attr = list(correspondings.keys())[0]
                val = correspondings[attr]['__value'][i]
                new_config = update_attribute_in_config(new_config, attr, val)
                new_explanation = add_explanation(new_explanation, attr, val)

            has_children = False
            for key in filter(lambda k: not k.startswith('__'), tree.keys()):
                has_children = True
                yield from extract_configs_from_attribute_tree(
                    tree[key], key, new_config, explanation=new_explanation)

            if not has_children:
                yield new_config, new_explanation

    for key in hyperparam_config.keys():
        yield from extract_configs_from_attribute_tree(hyperparam_config[key], key, base_config, explanation='')


def construct_task_name(explanation):

    name = ''

    def cond_query(label, test_query, true_val, false_val):
        if (label + "=" + test_query) in explanation:
            return true_val
        else:
            return false_val

    def extract_query(label, addition='', postval='_'):
        l = label + '='
        if l in explanation:
            index = explanation.index(l) + len(l)
            exp_begin = explanation[index:]
            if '-__-' in exp_begin:
                exp_end = exp_begin.index('-__-')
                val = exp_begin[:exp_end]
            else:
                val = exp_begin
            return addition + val + postval
        return addition + '???' + postval

    if 'mup_--_load_base_shapes' in explanation:
        name += extract_query('mup_--_load_base_shapes', addition='prm')

    if 'model_--_classification_--_resnet_--_width_multiplier' in explanation:
        name += extract_query('model_--_classification_--_resnet_--_width_multiplier', addition='wm')

    if 'model_--_classification_--_resnet_--_growth_factor' in explanation:
        name += extract_query('model_--_classification_--_resnet_--_growth_factor', addition='gf')

    if 'scheduler_--_cosine_--_max_lr' in explanation:
        name += extract_query('scheduler_--_cosine_--_max_lr', addition='mlr')

    if 'scheduler_--_cosine_--_min_lr' in explanation:
        name += extract_query('scheduler_--_cosine_--_min_lr', addition='ilr')

    if 'common_--_seed' in explanation:
        name += extract_query('common_--_seed', addition='seed')  # seed

    if 'optim_--_weight_decay' in explanation:
        name += extract_query('optim_--_weight_decay', addition='wd')

    return name


if __name__ == '__main__':

    print('Warning! Should be run inside *root* directory of the project.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_config_path', default='', help="Path to the base config file")
    parser.add_argument('--hyperparams_config_path', default='', help='Hyperparameters to choose from')
    parser.add_argument('--allocation_code', default='', help='Script allocation code')
    parser.add_argument('--user_mail', default='', help='Script user mail')
    parser.add_argument('--repetition_num', default=5, type=int, help='Script user mail')
    parser.add_argument('--cycle_time_in_hours', default=2, type=float, help='Script user mail')
    parser.add_argument('--pbs', dest='pbs', action='store_true')
    parser.add_argument('--sbatch', dest='pbs', action='store_false')
    parser.add_argument('--directory_modifier', default='', help='Storage directory of results modification')
    parser.add_argument('--force_random', dest='force_random', action='store_true')
    parser.add_argument('--cedar', dest='cedar', action='store_true')
    parser.add_argument('--random_mem', default=40, type=int, help='mem required for random jobs')
    parser.add_argument('--random_cpu', default=6, type=int, help='cpu # required for random jobs')
    parser.set_defaults(pbs=True, force_random=False, cedar=False)
    args = parser.parse_args()

    base_config = load_config(args.base_config_path)
    hyperparams_config = load_config(args.hyperparams_config_path)
    new_configs = list(generate_configs(base_config, hyperparams_config))

    today = date.today().strftime('%Y_%m_%d')
    root_path = 'jobs/' + today + args.directory_modifier
    path = Path(root_path)
    path.mkdir(parents=True)

    for config, explanation in new_configs:

        name = construct_task_name(explanation)  # todo: should not result in same thing for different configs
        job_path = path.joinpath(name)
        job_path.mkdir()
        config_path = str(job_path.joinpath('config.yml').absolute())

        if not args.pbs or args.cedar:
            raise NotImplementedError()

        # is_random = 'acquisition=random' in explanation
        if args.force_random:
            is_random = True
        else:
            is_random = False
        save_config(config, config_path)
        for i in range(args.repetition_num):
            script_path = str(job_path.joinpath('script_{}.sh'.format(i)).absolute())
            script = qsub_submit_script
            time_hrs = int(args.cycle_time_in_hours)
            # time_hrs = 8
            script = script.format(
                walltime=time_hrs,
                walldays=int(math.ceil(time_hrs / 24) + 1),
                ncpus=args.random_cpu if is_random else (24 if (args.pbs or args.cedar) else 48),
                # ncpus=6,
                # ncpus=(24 if args.pbs else 48),
                ngpus=1 if is_random else 4,
                gpu_type='v100l:' if args.cedar else '',
                # ngpus=1,
                # ngpus=4,
                gpu_mem='32gb',
                mem='{}gb'.format(args.random_mem) if is_random else ('187gb' if (args.pbs or args.cedar) else '490G'),
                # mem='100gb',
                # mem=('187gb' if args.pbs else '490G'),
                cuda_visible_devices_expr='' if is_random else 'export CUDA_VISIBLE_DEVICES=0,1,2,3',
                # cuda_visible_devices_expr='export CUDA_VISIBLE_DEVICES=0,1,2,3',
                allocation_code=args.allocation_code,
                user_mail=args.user_mail,
                config_path=config_path,
                run_dir=str(job_path.absolute()),
                name=name,
                job_index=i,
            )
            with open(script_path, 'w') as f:
                f.write(script)

    inp = input('Created all the files! Press any key to submit them...\n')

    for config, explanation in new_configs:
        name = construct_task_name(explanation)
        job_path = path.joinpath(name)
        config_path = str(job_path.joinpath('config.yml').absolute())
        for i in range(args.repetition_num):
            script_path = str(job_path.joinpath('script_{}.sh'.format(i)).absolute())
            if args.pbs:
                subprocess.check_call(['qsub', script_path])
            else:
                subprocess.check_call(['sbatch', script_path])
