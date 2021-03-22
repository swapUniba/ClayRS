import setuptools
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='orange_cb_recsys',
      version='0.3.9',
      author='Roberto Barile, Francesco Benedetti, Carlo Parisi, Mattia Patruno',
      install_requires=[
          'pandas==1.0.5',
          'PyYAML==5.3.1',
          'numpy==1.18.4',
          'gensim==3.8.3',
          'nltk==3.5',
          'babelpy==1.0.1',
          'mysql==0.0.2',
          'mysql-connector-python==8.0.20',
          'wikipedia2vec==1.0.4',
          'sklearn==0.0',
          'SPARQLWrapper==1.8.5',
          'textblob==0.15.3',
          'matplotlib==3.2.2',
          'pywsd==1.2.4',
          'wn==0.0.23',
          'networkx==2.5',
          'progressbar2==3.53.1',
          'whoosh==2.7.4'
      ],
      description='Python Framework for Content-Based Recommeder Systems',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/m3ttiw/orange_cb_recsys',
      packages=setuptools.find_packages(),
      python_requires='>=3.5'
      )
