"""
Scikit-Optimize, or `skopt`, is a simple and efficient library to
minimize (very) expensive and noisy black-box functions. It implements
several methods for sequential model-based optimization. `skopt` is reusable
in many contexts and accessible.

[![Build Status](https://travis-ci.org/mimba/scikit-optimize.svg?branch=master)](https://travis-ci.org/mimba/scikit-optimize)

## Install

```
pip install scikit-optimize-w
```

## Getting started

Find the minimum of the noisy function `f(x)` over the range `-2 < x < 2`
with `skopt`:

```python
import numpy as np
from skopt import gp_minimize

def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            np.random.randn() * 0.1)

res = gp_minimize(f, [(-2.0, 2.0)])
```

For more read the [introduction to bayesian optimization](https://scikit-optimize.github.io/notebooks/bayesian-optimization.html) offerec by scikit-optimize
and the other [examples](https://github.com/mimba/scikit-optimize/tree/master/examples).

res = gp_minimize(f, [(-2.0, 2.0)])
```

For more read the [introduction to bayesian optimization](https://scikit-optimize.github.io/notebooks/bayesian-optimization.html)
and the other [examples](https://github.com/mimba/scikit-optimize/tree/master/examples).


## Development

The library is still experimental and under heavy development.

The development version can be installed through:

    git clone https://github.com/scikit-optimize/scikit-optimize.git
    cd scikit-optimize
    pip install -r requirements.txt
    python setup.py develop

Run the tests by executing `pytest` in the top level directory.
"""

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    __SKOPT_SETUP__
except NameError:
    __SKOPT_SETUP__ = False

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "2020.1.3"

if __SKOPT_SETUP__:
    import sys
    sys.stderr.write('Partial import of skopt during the build process.\n')
    # We are not importing the rest of scikit-optimize during the build
    # process, as it may not be compiled yet
else:
    import platform
    import struct
    from . import acquisition
    from . import benchmarks
    from . import callbacks
    from . import ext
    from . import learning
    from . import optimizer

    from . import space
    from . import sampler
    from .ext import weighted_validation
    from .optimizer import dummy_minimize
    from .optimizer import forest_minimize
    from .optimizer import gbrt_minimize
    from .optimizer import gp_minimize
    from .optimizer import Optimizer
    from .searchcv import WeightedBayesSearchCV
    from .space import Space
    from .utils import dump
    from .utils import expected_minimum
    from .utils import expected_minimum_random_sampling
    from .utils import load
    __all__ = (
        "acquisition",
        "benchmarks",
        "callbacks",
        "ext",
        "weighted_validation",
        "learning",
        "optimizer",
        "plots",
        "sampler",
        "space",
        "gp_minimize",
        "dummy_minimize",
        "forest_minimize",
        "gbrt_minimize",
        "Optimizer",
        "dump",
        "load",
        "expected_minimum",
        "WeightedBayesSearchCV",
        "Space"
    )
    IS_PYPY = platform.python_implementation() == 'PyPy'
    _IS_32BIT = 8 * struct.calcsize("P") == 32
