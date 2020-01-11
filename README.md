# GPJ.jl

[![Build Status](https://travis-ci.com/TuringLang/GPJ.jl.svg?branch=master)](https://travis-ci.org/TuringLang/GPJ.jl)

A GP library in Julia.

Currently this package interfaces with [GPFlow](https://github.com/GPflow/GPflow).

## Getting Started

To use GPJ.jl, you need to install Julia first and then install GPJ.jl.

The following will install the latest version of GPJ.jl while inside Julia’s package manager (press `]` from the REPL):
```julia
    add https://github.com/TuringLang/GPJ.jl/
    build GPJ   # this should install the GPFlow python package and its dependencies.
```

## Plan

1. Add Julia interface for [GPFlow](https://github.com/GPflow/GPflow) and [GPt](https://github.com/ialong/GPt)
2. Add Julia interface for [GPyTorch](https://github.com/cornellius-gp/gpytorch)
3. Pure Julia support for GP by porting [GPML](https://github.com/alshedivat/gpml) (see e.g. [GPKit.jl](https://github.com/adriancu/GPkit.jl), [Stheno.jl](https://github.com/willtebbutt/Stheno.jl), [theogf/AugmentedGaussianProcesses.jl](https://github.com/theogf/AugmentedGaussianProcesses.jl) and [STOR-i/GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl))  


## Related papers

### GPFlow:

1. Matthews, De G., G. Alexander, Mark Van Der Wilk, Tom Nickson, Keisuke Fujii, Alexis Boukouvalas, Pablo León-Villagrá, Zoubin Ghahramani, and James Hensman. "GPflow: A Gaussian process library using TensorFlow." The Journal of Machine Learning Research 18, no. 1 (2017): 1299-1304.


### GPyTorch:

2. Wang, Ke Alexander, Geoff Pleiss, Jacob R. Gardner, Stephen Tyree, Kilian Q. Weinberger, and Andrew Gordon Wilson. "Exact Gaussian Processes on a Million Data Points." arXiv preprint arXiv:1903.08114 (2019).

3. Gardner, Jacob R., Geoff Pleiss, David Bindel, Kilian Q. Weinberger, and Andrew Gordon Wilson. ” GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration.” In NeurIPS (2018).


General intro paper on sparse variational inference in GP:

4. Quiñonero-Candela, Joaquin, and Carl Edward Rasmussen. "A unifying view of sparse approximate Gaussian process regression." Journal of Machine Learning Research 6, no. Dec (2005): 1939-1959.

5. Bauer, Matthias, Mark van der Wilk, and Carl Edward Rasmussen. "Understanding probabilistic sparse Gaussian process approximations." In Advances in neural information processing systems, pp. 1533-1541. 2016. (GPFlow implements several algorithms here)

Also the GP book

6. Gaussian Processes for Machine Learning,
Carl Edward Rasmussen and Christopher K. I. Williams, MIT Press (2006). 
