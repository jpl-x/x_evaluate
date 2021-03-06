from distutils.core import setup

setup(
    name='x-evaluate',
    version='1.0.0',
    scripts=['test/evaluate.py'],
    packages=['x_evaluate', 'x_evaluate.rpg_tracking_analysis'],
    package_dir={'x_evaluate': 'src/x_evaluate'},
    install_requires=['numpy>=1.19.2',
                      'matplotlib>=3.3.4',
                      'envyaml>=1.7',
                      'evo>=1.13.4',
                      'gitpython>=3.1.14',
                      'pyyaml>=5.4.1']
)
