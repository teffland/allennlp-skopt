""" Optimize a base allennlp configuration with a gaussian process by providing
a hyperparam search file.
"""
from argparse import ArgumentParser
import os
import numpy as np
import json
import _jsonnet
import subprocess
from datetime import datetime

from collections import OrderedDict, defaultdict

import skopt
from skopt.space.space import Categorical, Integer, Real, Space
from skopt.utils import normalize_dimensions

from allennlp.common.params import parse_overrides, with_fallback
from allenopt.util import *
from allenopt.plot import *

logger = init_logger(__name__, logging.DEBUG)

def parse_args(args=[]):
    parser = ArgumentParser(description="Optimize allennlp model hyperparams with random, gaussian process, or tree-based process")
    parser.add_argument('base_config_path', help="Base config path")
    parser.add_argument('search_config_path', help="Search space config path")
    parser.add_argument('--include-package', help='Source package to pass to allennlp')
    parser.add_argument('-s', '--serialization-dir', type=str, help="Base directory to save trials in." )
    parser.add_argument('-o', '--overrides', type=str, default=None, help="If provided, we will override the base config with these")
    parser.add_argument('-e', '--evaluate-on-test', type=str, default=None, help="If provided, we will evaluate the best model on this test set.")
    parser.add_argument('-n', '--n-calls', type=int, default=10, help="Number of trials")
    parser.add_argument('-r', '--random-seed', type=int, default=None, help="Set a random state.")
    parser.add_argument('-m', '--mode', type=str, default='gp', choices=['random', 'tree', 'gp'], help="Minimizer type. 'gp' is gaussian process, 'random' is random search, 'tree' is extra trees search.")
    parser.add_argument('--n-random-starts', type=int, default=1, help="If provided, seed process with n random function evals in addition to the defaul x0")
    parser.add_argument('--xi', type=float, default=0.01, help="Exploration/expoitation param")
    parser.add_argument('--kappa', type=float, default=1.96, help="Exploration/expoitation param")
    parser.add_argument('--no-delete-worse', action='store_true', help='By default we delete heavy ".th" and ".gz" files for worse trials as we go to save disk space. This disables that.')
    return parser.parse_args(args) if args else parser.parse_args()


def run(args):
    # Create base serialization dir
    if not os.path.exists(args.serialization_dir):
        os.makedirs(args.serialization_dir)

    # Read in search configuration and create the blackbox function to optimize
    f, dimensions, x0, trial_paths, delete_worse_files_cb = setup(args)
    n_random_starts = max(1,args.n_random_starts) if x0 is None else args.n_random_starts
    callback = None if args.no_delete_worse else delete_worse_files_cb

    # Run the actual optimization
    if args.mode == 'gp':
        results = skopt.gp_minimize(
            f, dimensions,
            x0=x0,
            n_calls=args.n_calls,
            n_random_starts=n_random_starts,
            random_state=args.random_seed,
            verbose=True,
            acq_optimizer='sampling',
            xi=args.xi,
            kappa=args.kappa,
            callback=callback,
        )
    elif args.mode == 'random':
        results = skopt.dummy_minimize(
            f, dimensions,
            x0=x0,
            n_calls=args.n_calls,
            random_state=args.random_seed,
            verbose=True,
            callback=callback,
        )

    elif args.mode == 'tree':
        results = skopt.forest_minimize(
            f, dimensions,
            x0=x0,
            n_calls=args.n_calls,
            n_random_starts=n_random_starts,
            random_state=args.random_seed,
            verbose=True,
            xi=args.xi,
            kappa=args.kappa,
            callback=callback,
        )


    # Maybe evaluate the best model on the test dataset
    if args.evaluate_on_test:
        logger.info('EVALUATE ON TEST')
        evaluate_on_test(args, results, trial_paths)

    # Save a bunch of visualizations of the search process
    logger.info('PLOTTING RESULTS')
    plot_results(args.serialization_dir, results)

    logger.info('ALL DONE')


def setup(args):
    """ Create the blackbox function to optimize.

    This is a complex function that wraps the true parameter setting and training
    in subprocess calls to allennlp.
    """
    base_config = json.loads(_jsonnet.evaluate_file(args.base_config_path))
    search_config = json.loads(_jsonnet.evaluate_file(args.search_config_path))

    # Override the base config with any arg overrides
    overrides = parse_overrides(args.overrides)
    base_config = with_fallback(preferred=overrides, fallback=base_config)

    # Flatten configs and get shorthand mappings
    flat_base_config = flatten(base_config)
    flat_search_config = flatten(search_config)
    shorthands = get_shorthands(flat_search_config)

    # Extract any variable dimensions and the mapping to their keys
    search_space = extract_search_space(flat_search_config)
    lambdas = extract_lambdas(flat_search_config)
    dimensions = list(search_space.values())

    # We no longer use the base config as an initial point because the base config
    # needs to be minimal -- cannot contain fields which aren't used by certain hp
    # configurations since overrides cannot "delete" a field in the base config.
    x0 = None  # get_x0(flat_base_config, search_space)

    trial_num = 0
    trial_paths = dict()

    # Construct f
    def f(x):
        nonlocal trial_num
        nonlocal trial_paths

        # Map the x to the config keys that need updated
        newx = []
        for d,p in zip(dimensions, x):
            print(d.name, d, p, type(p))
            if 'numpy' in str(type(p)):
                p = p.item()
            newx.append(p)
        x = newx
        overrides = skopt.utils.point_asdict(search_space, x)
        overrides = fill_search_constants(overrides, flat_search_config)
        overrides = restrict_type_overrides(overrides, flat_search_config)

        print(f'Overrides after fill and restrict: {json.dumps(overrides, indent=2)}')

        # Construct the trial serialization path
        trial_str = construct_trial_name(overrides, shorthands, trial_num)
        trial_path = os.path.join(args.serialization_dir, trial_str)
        trial_paths[trial_num] = trial_path

        # Construct the overrides string
        processed_overrides = format_overrides(overrides, lambdas)
        print(f'Sampled point: {json.dumps(processed_overrides, indent=2)}')
        override_str = json.dumps(processed_overrides, indent=None)

        # Run Allennlp train subprocess
        cmd = f"allennlp train {args.base_config_path} -f -s {trial_path} -o '{override_str}' --file-friendly-logging --include-package {args.include_package}"
        print(f'CMD: {cmd}')
        try:
            subprocess.check_call(cmd, shell=True)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise e

        trial_num += 1

        # Retrieve the best validation metric and return that value
        metrics = json.load(open(os.path.join(trial_path, 'metrics.json')))
        validation_metric = base_config['trainer']['validation_metric']
        negate = validation_metric.startswith('+')
        validation_metric = validation_metric.lstrip('+-')
        y = metrics[f'best_validation_{validation_metric}']
        if negate:
            y = -y

        return y

    # Construct a callback which maintains only the best weights/archive
    def delete_worse_files_cb(results):
        """ Remove .th and .gz files for any trials that aren't the best so far.
        """
        nonlocal trial_num
        nonlocal trial_paths
        logger.info(f'DELETE WORSE FILES, trial num:{trial_num}')

        best_trial_num = np.argmin(results.func_vals).item()
        logger.info(f'Func values: {results.func_vals},  best is {best_trial_num} with path {trial_paths[best_trial_num]}')
        for i in range(trial_num):
            if i != best_trial_num:
                logger.info(f'Deleting .th and .gz files at {trial_paths[i]}')
                th_path = os.path.join(trial_paths[i], '*.th')
                gz_path = os.path.join(trial_paths[i], '*.gz')
                cmd = f"rm -f {th_path} && rm -f {gz_path}"
                try:
                    subprocess.check_call(cmd, shell=True)
                except Exception as e:
                    logger.error(e, exc_info=True)
                    raise e

    return f, dimensions, x0, trial_paths, delete_worse_files_cb



def evaluate_on_test(args, results, trial_paths):
    """ Look at all models in serialization dir for the argmaximizer
    of the 'best_validation_metric', then evaluate that model on the test set.
    """
    # Find the best trial model
    best_trial_num = np.argmin(results.func_vals).item()
    best_trial_path = trial_paths[best_trial_num]
    model_path = os.path.join(best_trial_path, 'model.tar.gz')

    # Evaluate that model on the test dataset, dumping to best_trial_test_results.jsons
    output_path = os.path.join(args.serialization_dir, 'best_trial_test_metrics.json')
    cuda_device = json.loads(_jsonnet.evaluate_file(args.base_config_path))['trainer'].get('cuda_device', -1)
    cmd = f"allennlp evaluate {model_path} {args.evaluate_on_test} --output-file {output_path} --cuda-device {cuda_device} --include-package {args.include_package}"
    logger.info(f'EVALUATE CMD: {cmd}')
    try:
        subprocess.check_call(cmd, shell=True)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e

    # Open the results and add the path of the best model so we know who won.
    test_metrics = json.load(open(output_path))
    test_metrics['best_trial_path'] = best_trial_path
    logger.info(f'Best trial path was {best_trial_path} with test metrics:{json.dumps(test_metrics, indent=2)}')
    with open(output_path, 'w') as outf:
        json.dump(test_metrics, outf)



if __name__ == '__main__':
    args = parse_args()
    run(args)
