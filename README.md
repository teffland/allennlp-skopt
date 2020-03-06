# allennlp-skopt
Bayesian hyperparameter search for models in [allennlp]() using [skopt]().


Allennlp is great, but one thing that's missing is intelligent hyperparameter
tuning for models. This small package provides a wrapper around allennlp training
calls that can be plugged in to skopt to perform bayesian hyperparameter search.

It combines the flexibility of allennlp config files and overrides with the power
of scikit-optimize sequential model-based optimization. All you need to do is
provide a base allennlp training config file together with a search space config file
(this library).

This search space config file is lightly marked-up subset of the base config.
We provide a simple but flexible markup language for specifying the
dimensions of the search (along with some other features below.)


# Features
* Bayesian hyperparam search just by specifying a search-space config file -- better models with fewer runs.
* Hyperparam search in high-dimensional mixed data-type spaces using skopt.
* Easy and complete model configuration, training with allennlp.
* Automatic naming and management of trial runs based on sampled hps.
* Advanced hyperparameter logic with type-specific dimensions and deterministic
 dimensions.

# Installation


# Tutorial

A hyperparam search has two components:
1. An allennlp base config jsonnet file, stripped down to all non-varying hyperparameters
2. A hp search base config jsonnet file, specifynig the search space for any varying hyperparameters.

## Specifying the search space

Hyperparameter search on complex architectures and optimization setups can be difficult
to specify, often requiring bespoke scripts. We instead leverage the flexibility
and utility of allennlp's config files with an extended syntax that allows specification
of search dimension ranges and choices.


Hyperparam search configuration files should mirror the structure of the base
config, stripped down only to the fields which should be explored (and any others needed
for allennlp's --override argument as constants, more on this later.)

### Search Dimensions

The fields we want searched over should be wrapped in a list, giving either the range
(for int and real dimensions) or choices (categorical). If the list is two numerical
values, it's interpreted as a range (order doesn't matter). If it's three or more,
or the values are strings or special types (`null`, `true`, `false`), then it's interpreted as categorical.

For example we may have:

```js
    "trainer": {
      "optimizer": {
        "type": ["sgd", "adam", "adamw"],  // categorical choices of optimizer
        "lr": [1e-5,1e-3],                 // real valued range
      }
    }
```

**Protip:** If you want a categorical choice of two numbers, then duplicate
one of them -- for categories we deduplicate the lists (values won't be
cast to strings).

For example we might have:

```js
    "trainer": {
      "optimizer": {
        "type": ["sgd", "adam", "adamw"],
        "lr": [1e-3,1e-5,1e-5]             // 1e-3 and 1e-5 are only choices
      }
    }
```

### Constant Dimensions

Singleton options can be passed as constants, not wrapped in lists. In this case,
this param will override the base config and be included in the serialization string
but will not actually be varied over/modeled by the hp minimizer.

```js
    "trainer": {
      "optimizer": {
        "type": "sgd",          // always override to "sgd", don't actually model.
        "momentum": [0.9,0.99]
      }
    }
```

Additionally, length-one lists are unwrapped, which allows for lists to be passed as constants:

```js
    "m": {
      "optimizer": {
        "type": "adam",          
        "betas": [[0.9,0.99]]   // these are kept constant as [0.9,0.99]
      }
    }
```

This is especially useful when specifying type-specific settings via conditional dimensions, which leads us too...

### Conditional Dimensions

Sometimes we want to search over multiple "type"s of objects, such as different
optimizers, and these can (and often do) have different parameters.  Understandably,
allennlp gets mad when you try to pass inappropriate parameters.

To handle differences in arguments based on the object "type",
the parameter can be conditionally specified by prepending the key with the signature:
 `"type1|type2__<param_name>"`. This parameter will then only be passed to allennlp
 when `"type"` in `{"type1", "type2"}`.  


 This is a powerful tool and now we can do things like:

```js
    "trainer": {
      "optimizer": {
        "type": ["sgd", "adam", "adamw"],
        "sgd__momentum": [0.9,0.99],          // this is only passed to sgd
        "adamw__weight_decay": [1e-4,1e-2],   // this is only passed to the adamw
        "adam|adamw__epsilon": [1e-9, 1e-7]   // this is only passed to the adam and adamw
      }
    }
```

As we alluded to above, this can be combined with constant dimensions to specify
arguments which are unvarying but only apply to a subset of the varying types.

For example:

```js
    "trainer": {
      "optimizer": {
        "type": ["sgd", "adam", "adamw"],
        "adam__eps": 1e-8                    // this is only passed to adam but left constant
      }
    }
```

### Deterministic Dimensions

Finally, sometimes you have dimensions that should only vary deterministically
with other dimensions you are searching over.  We solve this using "lambda" dimensions
specified in strings. The specified function will be given the sampled search configuration
dictionary as input.

For example. maybe we want the learning rate schedule to depend on other optimizer parameters,
such as the number of epochs, which itself is part of the hyperparamter search:

```js
    "trainer": {
      "num_epochs": [5,50],   // this will vary
      "optimizer": {
        "type": "sgd",
        "lr": 1e-3
      }
      "learning_rate_scheduler": {
        "type": "step",
        // this deterministically defines gamma to anneal the learning rate to 0 over the course of training
        "gamma": "lambda c:(c['trainer']['optimizer']['lr']/c['trainer']['num_epochs']"
      }
    }
```

**Note:** this dictionary does NOT include any parameter not specified in the _search space_ file,
such as those specified in the original base config.

For example, if the _base_ config file in the above example had:

```js
    "trainer": {
      "optimizer": {
        "type": "sgd",
        "lr": 1e-3
      }
    }
```

then we may be tempted to write the search space file as:

```js
    "trainer": {
      "num_epochs": [5,50],   // this will vary
      "learning_rate_scheduler": {
        "type": "step",
        // this breaks because ['optimizer']['lr'] isnt in the search space file
        "gamma": "lambda c:(c['trainer']['optimizer']['lr']/c['trainer']['num_epochs']"
      }
    }
```

**Sidenote:** As of now, I think lambda functions are executed in a non-deterministic
order, so chaining them together isn't possible. Not sure it's useful (would be kind of cool though I guess.)


### Fine-grained Control of the Dimensions

`skopt` offers some additional, more fine-grained control of the search dimensions.
Currently these transformations of the space before sampling, prior distributions on the sample space,
and base of log when the prior is "log-uniform".

These can be specified for dimensions using the following special reserved suffixes.
1. `<param>__TRANSFORM`: transform for the dimension
2. `<param>__PRIOR`: sampling distribution for the dimension
3. `<param>__BASE`: base of the log when `__PRIOR` is `log-uniform`

One example (and common use-case) is when we want to sample in the log-domain:

```js
  "trainer": {
    "optimizer": {
      "lr": [1e-6,1e-3],
      "lr__PRIOR": "log-uniform"  // samples base10 magnitude of learning rate uniformly
    }
  }
```

Or for batch sizes:
```js
  "data_loader": {
    "batch_size": [4,64],
    "batch_size__BASE": 2               // samples base2 magnitude of learning rate uniformly
    "batch_size__PRIOR": "log-uniform"  //
  }
```

**IMPORTANT:** reserved suffixes use prefixes that _exactly_ match the dimension keys
they are modifying, including any shorthand annotations described in the following section.


### Sample Naming and Parameter Shorthanding

When actually running the experiments, we have to give them names so allennlp
can save them. We've elected to do this by encoding the sampled configuration
into a string as the model name after the metadata. This allows reading off the
training settings directly from the name, and is particularly powerful when used
together with tensorboard's regex filtering facilities.

The problem with this though is that allennlp config files are verbose and
naively using the fully flattened search config results in ridiculously long strings.
To remedy this, we allow for specifying field "shorthand" annotations by appending
a colon and the shorthand, as in `"<super_verbose_param_name>:<shrt_name>"`.

An example helps. Say we have:

```js
  "trainer": {
    "optimizer": {
      "type": ["sgd", "adam"],
      "lr": [1e-6,1e-3],
    }
  }
```

and the search samples the point `{"type":"sgd", "lr':1e-6"}`. Then the trial name
would be the

```
  trainer-optimizer-type=sgd_trainer-optimizer-lr=1e-6
```

which is ridicuously long. For realistic searches it can easily exceed the max unix file name length.
With some simple shorthand annotations though, we can modify our config:

```js
  "trainer:t": {                     // shorthand 'trainer' to 't'
    "optimizer:o": {                 // likewise
      "type:mode": ["sgd", "adam"],  // can be used for just renaming if that's your thing
      "lr": [1e-6,1e-3],
    }
  }
```

and with the sample search point our trial name would be
```
  t-o-mode=sgd_t-o-lr=1e-6
```

**Note:** shorthands can be the empty string. If the shorthand for a "leaf" key
is the empty string, then we omit that branch of the config tree in the name altogether.

Continuing with the above example:

```js
  "trainer:t": {                     
    "optimizer:": {                 // will not be in the nested name
      "type:": "sgd",               // whole branch wont be in the name
      "lr": [e-6,1e-3],
    }
  }
```

we'd just get
```
  t-lr=1e-6
```

These features can be really nice for reducing the complexity of your search
naming to something thats extremely informative but manageable.


**So actually** we lied, the above naming convention is only a part of the full name,
calling it `<config_str>`, the full name is then:

```
  <datetime>_hp-trial-<trial_num>_<config_string>_<trial_num>
```


## Running the optimizer

Once you've setup your search file, you optimize it with the following signature:

```bash
allenopt.optimize

usage: optimize.py [-h] [--include-package INCLUDE_PACKAGE]
                   [-s SERIALIZATION_DIR] [-e EVALUATE_ON_TEST] [-n N_CALLS]
                   [-r RANDOM_SEED] [-m MODE]
                   [--n-random-starts N_RANDOM_STARTS] [--xi XI]
                   [--kappa KAPPA] [--no-delete-worse]
                   base_config_path search_config_path
```

And that's all you need to do! Now the script will run the specified skopt
procedure using allennlp by searching in your provided search space file.
It will create a bunch of seriralization directories within the base `-s <serialzation_dir>`.
You can additionally have the best found model (based on held-out validation metric in allennlp base config)
evaluate on a test set with `-e`.

An example command, where we run a gaussian process optimization for some given setup,
with 20 trials would be:

```bash
allenopt.optimize\
 --include-package <my_allennlp_package>\
 -s <a directory path to save to>\
 -n 20\
 -m gp\
 -r 42\
 my_base_allennlp_train_config.jsonnet\
 my_search_config.jsonnet
```

Check out the [examples](examples) for some complete examples.

### Command Arg Details
```bash
allenopt.optimize -h

Optimize allennlp model hyperparams with random, gaussian process, or tree-
based process

positional arguments:
  base_config_path      Base config path
  search_config_path    Search space config path

optional arguments:
  -h, --help            show this help message and exit
  --include-package INCLUDE_PACKAGE
                        Source package to pass to allennlp
  -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                        Base directory to save trials in.
  -e EVALUATE_ON_TEST, --evaluate-on-test EVALUATE_ON_TEST
                        If provided, we will evaluate the best model on this
                        test set.
  -n N_CALLS, --n-calls N_CALLS
                        Number of trials
  -r RANDOM_SEED, --random-seed RANDOM_SEED
                        Set a random state.
  -m {random,tree,gp}, --mode {random,tree,gp}
                        Minimizer type. 'gp' is gaussian process, 'random' is
                        random search, 'tree' is extra trees search.
  --n-random-starts N_RANDOM_STARTS
                        If provided, seed process with n random function evals
                        in addition to the defaul x0
  --xi XI               Exploration/expoitation param
  --kappa KAPPA         Exploration/expoitation param
  --no-delete-worse     By default we delete heavy files for worse trials as
                        we go. This disables that.
```


# TODO:
* [X] Add evaluate on test for best validation model
* [ ] Pipe actual allennlp run logs to separate files.
* [X] Remove all weights and models except the best.th and model.tar.gz from the
 best held out model, do this live by tracking best values so far -- significant space decrease.
 put this in a callback so we can do it on the fly.
* [X] Save the results object and save plots
  * [X] Plot convergence
  * [X] Plot objective
  * [X] Plot samples
