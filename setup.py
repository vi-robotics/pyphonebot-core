#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyphonebot_core",
    version="0.0.1",
    author="Yoonyoung Cho and Maximilian Schommer",
    description="Core pyphonebot software with minimal dependencies.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vi-robotics/pyphonebot-core",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'cho_util',
        'numpy',
        'networkx'
    ],
    extras_require={
    },
    python_requires='>=3.6',
)
