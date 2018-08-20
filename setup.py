#!/usr/bin/env python

from setuptools import setup, find_packages

REQUIREMENTS = [
    "numpy",
    "networkx"
]

# extra dependencies
EXTRAS = {
    "dev": [
        "pylint",
        "bandit",
        "sphinx",
        "sphinx_rtd_theme",
        "numpydoc",
        "nbsphinx"
    ],
    "gui": [
        "PyQt5"
    ]
}

setup(
    name="cluster",
    use_scm_version=True,
    description="Geometric constraint solver using modified cluster rewriting",
    author="Sean Leavey",
    author_email="software@attackllama.com",
    url="https://github.com/SeanDS/cluster",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS,
    setup_requires=['setuptools_scm'],
    license="GPLv3",
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6"
    ]
)
