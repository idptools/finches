from setuptools import setup, find_packages

setup(
    name='finches',
    packages=find_packages(),
    install_requires=[
        'soursop>=0.2.4',
        'afrc>=0.3.4',
        'pandas',
        'numpy',
        'scipy',
        # other dependencies
    ],
)
