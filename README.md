# allennlp-skopt
Bayesian hyperparameter search for models in [allennlp]() using [skopt]().


Allennlp is great, but one thing that's missing is intelligent hyperparameter
tuning for models. This small package provides a wrapper around allennlp training
calls that can be plugged in to skopt to perform bayesian hyperparameter search.

It combines the power of allennlp config files with the ease of hp search with skopt.
Further it defines a flexible markup language for specifying the search by mirroring
allennlp's configuration format.

# Features
* Bayesian hyperparam search just by specifying a search-space config file -- better models with fewer runs.
* Hyperparam search in high-dimensional mixed data-type spaces using skopt.
* Automatic naming and management of trial runs based on sampled hps.
* Conditional hyperparameter logic with alternations and lambda functions.


# Installation


# Tutorial

A hyperparam search has two components:
1. An allennlp base config jsonnet file, stripped down to all non-varying hyperparameters
2. A hp search base config jsonnet file, specifynig the search space for any varying hyperparameters.

# Search space markup files

Hyperparameter search on complex architectures and optimization setups can be difficult
to specify, often requiring custom scripts. We instead leverage the flexibility
and utility of allennlp's config files with an extended syntax that allows specification
of conditional logic and namespace shortcutting.







# An example
