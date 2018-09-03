try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import versioneer

setup(name='scikit-optimize-w',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Fork of scikit-optimize - Sequential model-based optimization toolbox - extended with sample weights',
      long_description=open('README.rst').read(),
      url='https://github.com/mimba/scikit-optimize',
      license='BSD',
      author='The scikit-optimize contributors and mimba',
      packages=['skopt', 'skopt.ext', 'skopt.learning', 'skopt.optimizer', 'skopt.space',
                'skopt.learning.gaussian_process'],
      install_requires=['pyaml', 'numpy', 'scipy>=0.14.0',
                        'scikit-learn>=0.19.2'],
      extras_require={
          'plots': ["matplotlib"]
      })
