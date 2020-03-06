# allennlp-skopt
Bayesian hyperparameter search for models in [allennlp]() using [skopt]().


Allennlp is great, but one thing that's missing is intelligent hyperparameter
tuning for models. This small package provides a wrapper around allennlp training
calls that can be plugged in to skopt to perform bayesian hyperparameter search.

It combines the flexibility of allennlp config files and overrides with the power
of scikit-optimize sequential model-based optimization. All you need is
to specify a search space config file which mirroring a base allennlp train config.

Further we've defined a simple but flexible markup language for specifying the
dimensions of the search (along with some other features below.)


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
of conditional logic and namespace shortcutting.

### Syntax

Hyperparam search configuration files should mirror the structure of the base
config stripped only to the fields which should be explored and any others needed
for allennlp's --override argument as constants.

#### Search Dimensions

The fields we want searched over should be wrapped in a list, giving either the range
(for int and real dimensions) or choices (categorical). If the list is two numerical
values, it's interpreted as a range (order doesn't matter). If it's three or more,
or the values are strings or special types (null and bools), then it's interpreted as categorical.

For example we may have:

```json
    "trainer": {
      "optimizer": {
        "type": ["sgd", "adam", "adamw"],  # categorical choices of optimizer
        "lr": [1e-5,1e-3],                 # real valued range
        ""
      }
    }
```

**Protip:** If you want a categorical choice of two numbers, then duplicate
one of them -- for categories we deduplicate the lists (values won't be
cast to strings).

For example we might have:

```json
    "trainer": {
      "optimizer": {
        "type": ["sgd", "adam", "adamw"],
        "lr": [1e-3,1e-5,1e-5]  # 1e-3 and 1e-5 are only choices
      }
    }
```

#### Constant Dimensions

Singleton options can be passed as constants, not wrapped in lists. In this case,
this param will override the base config and be included in the serialization string
but will not actually be varied over/modeled by the hp minimizer.

```json
    "trainer": {
      "optimizer": {
        "type": "sgd",          # always override to "sgd", don't actually model.
        "momentum": [0.9,0.99]  
      }
    }
```

Additionally, length-one lists are unwrapped, which allows for lists to be passed as constants:

```json
    "m": {
      "optimizer": {
        "type": "adam",          
        "betas": [[0.9,0.99]]   # these are kept constant as [0.9,0.99]
      }
    }
```

This is especially useful when specifying type-specific settings via conditional dimensions, which leads us too...

#### Conditional Dimensions

Sometimes we want to search over multiple "type"s of objects, such as different
optimizers, and these can (and often do) have different parameters.  Understandably,
allennlp gets mad when you try to pass inappropriate parameters.

To handle differences in arguments based on the object "type",
the parameter can be conditionally specified by prepending the key with the signature:
 `"type1|type2__<param_name>"`. This parameter will then only be passed to allennlp
 when `"type"` in `{"type1", "type2"}`.  


 This is a powerful tool and now we can do things like:

```json
    "trainer": {
      "optimizer": {
        "type": ["sgd", "adam", "adamw"],
        "sgd__momentum": [0.9,0.99]          # this is only passed to sgd
        "adamw__weight_decay": [1e-4,1e-2]   # this is only passed to the adamw
        "adam|adamw__epsilon": [1e-9, 1e-7]  # this is only passed to the adam and adamw
      }
    }
```

As we alluded to above, this can be combined with constant dimensions to specify
arguments which are unvarying but only apply to a subset of the varying types.

For example:

```json
    "trainer": {
      "optimizer": {
        "type": ["sgd", "adam", "adamw"],
        "adam__eps": 1e-8                    # this is only passed to adam but left constant
      }
    }
```

#### Deterministic Dimensions

Finally, sometimes you have dimensions that should only vary deterministically
with other dimensions you are searching over.  We solve this using "lambda" dimensions
specified in strings. The specified function will be given the sampled search configuration
dictionary as input.

For example. maybe we want the learning rate schedule to depend on other optimizer parameters,
such as the number of epochs, which itself is part of the hyperparamter search:

```json
    "trainer": {
      "num_epochs": [5,50],   # this will vary
      "optimizer": {
        "type": "sgd",
        "lr": 1e-3
      }
      "learning_rate_scheduler": {
        "type": "step",
        # this deterministically defines gamma to anneal the learning rate to 0 over the course of training
        "gamma": "lambda c:(c['trainer']['optimizer']['lr']/c['trainer']['num_epochs']"
      }
    }
```

**Note:** this dictionary does NOT include any parameter not specified in the _search space_ file,
such as those specified in the original base config.

For example, if the _base_ config file in the above example had:

```json
    "trainer": {
      "optimizer": {
        "type": "sgd",
        "lr": 1e-3
      }
    }
```

then we may be tempted to write the search space file as:

```json
    "trainer": {
      "num_epochs": [5,50],   # this will vary
      "learning_rate_scheduler": {
        "type": "step",
        # this breaks because ['optimizer']['lr'] isnt in the search space file
        "gamma": "lambda c:(c['trainer']['optimizer']['lr']/c['trainer']['num_epochs']"
      }
    }
```

**Sidenote:** As of now, I think lambda functions are executed in a non-deterministic
order, so chaining them together isn't possible. Not sure it's useful (would be kind of cool though I guess.)


#### Fine-grained Control of the Dimensions

`skopt` offers some additional, more fine-grained control of the search dimensions.
Currently these transformations of the space before sampling, prior distributions on the sample space,
and base of log when the prior is "log-uniform".

These can be specified for dimensions using the following special reserved suffixes.
1. `<param>__TRANSFORM`: transform for the dimension
2. `<param>__PRIOR`: sampling distribution for the dimension
3. `<param>__BASE`: base of the log when `__PRIOR` is `log-uniform`

One example (and common use-case) is when we want to sample in the log-domain:

```json
  "trainer": {
    "optimizer": {
      "lr": [1e-6,1e-3],
      "lr__PRIOR": "log-uniform"  # samples base10 magnitude of learning rate uniformly
    }
  }
```

Or for batch sizes:
```json
  "data_loader": {
    "batch_size": [4,64],
    "batch_size__BASE": 2               # samples base2 magnitude of learning rate uniformly
    "batch_size__PRIOR": "log-uniform"  #
  }
```

**IMPORTANT:** reserved suffixes use prefixes that _exactly_ match the dimension keys
they are modifying, including any shorthand annotations described in the following section.


#### Sample Naming and Parameter Shorthanding

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

```json
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

```json
  "trainer:t": {                     # shorthand 'trainer' to 't'
    "optimizer:o": {                 # likewise
      "type:mode": ["sgd", "adam"],  # can be used for just renaming if that's your thing
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

```json
  "trainer:t": {                     
    "optimizer:": {                 # will not be in the nested name
      "type:": "sgd",               # whole branch wont be in the name
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


### Running the optimizer

Once you've setup your  # TODO
