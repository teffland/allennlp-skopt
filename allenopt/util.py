import os
import math
import numpy as np
import json
import _jsonnet
import subprocess
from datetime import datetime

from collections import OrderedDict, defaultdict

import skopt
from skopt.space.space import Categorical, Integer, Real, Space
from skopt.utils import normalize_dimensions

from allennlp.common.params import with_fallback


ROUTE_STR = '.'
RESERVED_SUFFIXES = { '__PRIOR', '__TRANSFORM', '__BASE' }

import sys
import logging
def init_logger(name, level):
    """ Make sure the logger flushes. """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    h = logging.StreamHandler(sys.stdout)
    h.flush = sys.stdout.flush
    logger.addHandler(h)

    return logger

logger = init_logger(__name__, logging.DEBUG)


def get_x0(flat_base_config, search_space):
    """ Extract a default point from the base configuration,
    replacing any invalid params.
    """
    x0 = [ flat_base_config.get(clean_nested_key(k), 'None') for k in search_space ]

    # Check x0 is in the space before running it
    # and coerce it into the search space with random samples where invalid
    dimensions = list(search_space.values())
    space = Space(normalize_dimensions(dimensions))
    for i, (p,d) in enumerate(zip(x0, space.dimensions)):
        if not p in d:
            sample = d.rvs()
            print(f"{p} not in dimension: {d} with name:{d.name}, setting to sample:{sample}")
            logger.info(f"{p} not in dimension: {d} with name:{d.name}, setting to sample:{sample}")
            x0[i] = sample
    print('x0', x0)
    print('space', space)
    return x0

def flatten(d):
    """ Flatten nested dictionary, connecting keys with route_str.
    """
    ret = dict()
    for k,v in d.items():
        if isinstance(v, dict):
            v = flatten(v)
            ret.update({f"{k}{ROUTE_STR}{h}":u for h,u in v.items()})
        else:
            ret[k] = v
    return ret

def unflatten(d, route_str=ROUTE_STR):
    """ Unflatten dictionary to nested dict, disconnecting on keys with by route_str.
    """
    ret = defaultdict(dict)
    for k,v in d.items():
        if ROUTE_STR in k:
            split = k.split(ROUTE_STR)
            k, subks =  split[0], ROUTE_STR.join(split[1:])
            ret[k][subks] = v
            if ROUTE_STR in subks:
                ret[k] = unflatten(ret[k])
        else:
            ret[k] = v
    return dict(ret)

def clean_key(k):
    """ Remove search space markup from key. """
    return k.split(':')[0].split('__')[-1]

def clean_nested_key(nk):
    return ROUTE_STR.join([clean_key(k) for k in nk.split(ROUTE_STR)])

def cleanup_markup(d):
    """ Remove search space markup from keys and unescape values.
    """
    ret = dict()
    for k,v in d.items():
        k = clean_key(k)
        if isinstance(v, dict):
            v = cleanup_markup(v)
            if 'type' in v and v['type'] == None:  # use type key to disable whole subfield
                v = None
        else:
            v = unescape_special(v)
        ret[k] = v
    return ret

def get_shorthands(flat_search_config):
    """ Convert flattened keys to shorthand notations for serialization str.
    """
    shorthands = {}
    for k in flat_search_config:
        if not any(k.endswith(s) for s in RESERVED_SUFFIXES):
            route = k.split(ROUTE_STR)
            shorthand = ''
            for subk in route[:-1]:
                shorthand += subk.split(':')[-1]+'-' if ':' in subk else ''
            shorthand += route[-1].split(':')[-1]

            shorthands[k] = shorthand
    return shorthands


def escape_special(v):
    """ Escape literal bools and None as strings.
    """
    if v is True or v is False or v is None:
        return str(v)
    else:
        return v

def unescape_special(v):
    """ Unescape string version of literals.
    """
    if v == 'True': return True
    elif v == 'False': return False
    elif v == 'None': return None
    else: return v

def extract_search_space(flat_search_config):
    """ Find the variable dimensions and convert them to a skopt search space.
    """
    search_space = OrderedDict()
    for k,v in flat_search_config.items():
        # Lists with more than one value are search dimensions
        if isinstance(v, list) and len(v) > 1:
            force_categorical = len(v) > 2

            # Dedupe the list, escaping specials, and sort smallest to largest
            ds = sorted({escape_special(u) for u in v})
            prior = flat_search_config.get(f'{k}__PRIOR', None)
            base = flat_search_config.get(f'{k}__BASE', 10)

            if force_categorical or isinstance(ds[0], str):
                transform = flat_search_config.get(f'{k}__TRANSFORM', 'onehot')
                dim = Categorical(ds, prior=prior, transform=transform, name=k)
            elif isinstance(ds[0], int):
                transform = flat_search_config.get(f'{k}__TRANSFORM', 'normalize')
                dim = Integer(*tuple(ds), prior=prior, transform=transform, base=base, name=k)
            elif isinstance(ds[0], float):
                transform = flat_search_config.get(f'{k}__TRANSFORM', 'normalize')
                dim = Real(*tuple(ds), prior=prior, transform=transform, base=base, name=k)

            search_space[k] = dim
    return search_space


def extract_lambdas(flat_search_config):
    """ Find any lambda functions for processing conditional hyperparams from a sampled point.
    """
    lambdas = OrderedDict()
    for k,v in flat_search_config.items():
        if isinstance(v, str) and v.startswith('lambda '):
            lambdas[k] = eval(v)
    return lambdas

def fill_search_constants(overrides, flat_search_config):
    """ Insert any items with constant values from the search config into overrides
    in case there are values needed by model when overriding but we aren't varying.
    """
    missing_keys = set(flat_search_config.keys()) - set(overrides.keys())
    overrides.update({k:flat_search_config[k] for k in missing_keys
                      if not any(k.endswith(s) for s in RESERVED_SUFFIXES)})
    return overrides

def restrict_type_overrides(overrides, flat_search_config):
    """ Remove any overrides that don't apply to provided type-constraints.
    """
    type_keys = [ k for k in overrides if k.split(ROUTE_STR)[-1].split(':')[0] == 'type' ]
    all_keys = list(overrides.keys())
    for tk in type_keys:
        route = ROUTE_STR.join(tk.split(ROUTE_STR)[:-1])
        t = overrides[tk]
        for k in all_keys:
            if k.startswith(route):
                other_leaf_key = k.split(ROUTE_STR)[-1]
                if '__' in other_leaf_key and not t in other_leaf_key.split('__')[0].split('|'):
                    overrides.pop(k)
                    print(f'Removed unapplicable key: {k} for type:{t}')

    return overrides

def format_overrides(overrides, lambdas, base_config, arg_overrides):
    """ Apply arg overrides and unflatten, then apply lambda dimensions

    (Note: We actually do it twice since the lambdas are indexed with flat keys)
    """
    # Fill samped overrides with base config, then fill with arg overrides
    format_ = lambda overrides:with_fallback(arg_overrides, with_fallback(cleanup_markup(unflatten(overrides)), base_config))
    nested_overrides = format_(overrides)
    for k,f in lambdas.items():
        overrides[k] = f(nested_overrides)
    return format_(overrides)


def construct_trial_name(overrides, shorthands, trial_num):
    """ Get a unique string for the hp configuration, omitting some special values.
    """
    date_str = datetime.strftime(datetime.now(), '%m-%d,%H:%M')
    s = f'{date_str}_hp-trial-{trial_num}_'
    keep = lambda k,v:not any(k.endswith(s) for s in RESERVED_SUFFIXES) and not str(v).startswith('lambda')
    str_format = lambda v:v.replace('__','-')
    for k,v in overrides.items():
        if keep(k,v):
            k = shorthands[k]
            if k and not k.endswith('-'):  # missing leaf shorthands are dropped -- allows specifying constants that are clog up the names
                if isinstance(v, float):
                    s += f'{k}={v:1.1e}_'
                elif isinstance(v, str):
                    s += f'{k}={str_format(v)}_'
                else:
                    s += f'{k}={v}_'
    s += f'hp-trial-{trial_num}'  # end with trial num also so we can see in tensorboard
    return s
