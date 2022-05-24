#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', 'numpy>=1.15.4', 'pandas>=0.23.4', 
                'tensorflow==2.6.4', 'keras>=2.2.4', 'scikit-learn>=0.20.2', 'matplotlib>=3.0.2',
                'seaborn>=0.9.0', 'scikit-image>=0.14.1', 'opencv-python>=4.1.0.25', 
                'multipledispatch>=0.6.0', 'tqdm>=4.28.1', 'pydicom>=1.2.2', 'setuptools>=39.1.0']

setup_requirements = [ ]
test_requirements = [ ]

setup(
    author="Ju Young Ahn",
    author_email='juyoung.ahn@snu.ac.kr',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6'
    ],
    description="From mammogram to breast cancer ",
    entry_points={
        'console_scripts': [
            'mammo2risk=mammo2risk.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mammo2risk',
    name='mammo2risk',
    packages=find_packages(include=['mammo2risk']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/JuYoungAhn/mammo2risk',
    version='0.1.0',
    zip_safe=False,
)
