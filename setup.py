import setuptools
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='orange_cb_recsys',
      version='0.3.9',
      author='Roberto Barile, Francesco Benedetti, Carlo Parisi, Mattia Patruno',
      install_requires=requirements,
      description='Python Framework for Content-Based Recommeder Systems',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/m3ttiw/orange_cb_recsys',
      packages=setuptools.find_packages(),
      python_requires='>=3.6'
      )
