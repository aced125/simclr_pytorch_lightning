#!/usr/bin/env python

"""The setup script."""
# fmt: off

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0',
    'pytorch-lightning',
    'torch',
    'torchvision',
    'omegaconf',
    'pytorch-lightning-bolts',
    'wandb'
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', 'black']

setup(
    author="Audrey Roy Greenfeld",
    author_email='audreyr@example.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="SimCLR implementation in PyTorch lightning",
    entry_points={
        'console_scripts': [
            'simclr_pytorch_lightning=simclr_pytorch_lightning.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='simclr_pytorch_lightning',
    name='simclr_pytorch_lightning',
    packages=find_packages(include=['simclr_pytorch_lightning', 'simclr_pytorch_lightning.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/aced125/simclr_pytorch_lightning',
    version='0.1.0',
    zip_safe=False,
)
