import setuptools
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = "0.5.1"

setup(name='clayrs',
      version=VERSION,
      license='GPL-3.0',
      author='Antonio Silletti, Elio Musacchio, Roberta Sallustio',
      install_requires=requirements,
      description='Complexly represent contents, build recommender systems, evaluate them. All in one place!',
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords=['recommender system', 'cbrs', 'evaluation', 'recsys'],
      url='https://github.com/swapUniba/ClayRS',
      include_package_data=True,
      packages=setuptools.find_packages(),
      python_requires='>=3.8',

      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Software Development :: Testing :: Unit'
      ]

      )
