from distutils.core import setup

setup(
    version='1.0.0',
    scripts=['test/evaluate.py'],
    packages=['x_evaluate'],
    package_dir={'x_evaluate': 'src/x_evaluate'},
    requires=['numpy>=1.19.2',
              'matplotlib>=3.3.4',
              'envyaml>=1.7',
              'evo>=1.13.4',
              'orjson>=3.5.1',
              'pyyaml>=5.4.1']
)
