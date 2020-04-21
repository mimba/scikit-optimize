try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
try:
    import builtins
except ImportError:
    print("Failed to import builtins")

# This is a bit (!) hackish: we are setting a global variable so that the
# main skopt __init__ can detect if it is being loaded by the setup
# routine
builtins.__SKOPT_SETUP__ = True

import skopt

VERSION = skopt.__version__

CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: BSD License',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']


setup(name='scikit-optimize-w',
      version=VERSION,
      description='Sequential model-based optimization toolbox.',
      long_description=open('README.rst').read(),
      url='https://github.com/mimba/scikit-optimize',
      license='BSD 3-clause',
      author='The scikit-optimize contributors',
      classifiers=CLASSIFIERS,
      packages=['skopt', 'skopt.learning', 'skopt.optimizer', 'skopt.space',
                'skopt.learning.gaussian_process', 'skopt.sampler'],
      install_requires=['joblib>=0.14.1', 'pyaml>=20.4.0', 'numpy>=1.18.3',
                        'scipy>=1.4.1',
                        'scikit-learn>=0.22.2.post1'],
      extras_require={
        'plots':  ["matplotlib>=3.2.1"]
        }

      )
