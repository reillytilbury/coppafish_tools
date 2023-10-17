import os
from setuptools import setup, find_packages

with open("coppafish-tools/_version.py", "r") as f:
    exec(f.read())

with open("README.md", "r") as f:
    long_desc = f.read()

packages = [folder for folder in find_packages() if folder[-5:] != '.test']  # Get rid of test packages

setup(
    name='coppafish-tools',
    version=__version__,
    description='coppaFISH software for Python',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    author='Josh Duffield',
    author_email='jduffield65@gmail.com',
    maintainer='Reilly Tilbury',
    maintainer_email='reillytilbury@gmail.com',
    license='MIT',
    python_requires='>=3.8, <3.11',
    url='https://jduffield65.github.io/coppafish/',
    packages=packages,
    install_requires=['coppafish[plotting,optimised]@git+https://github.com/reillytilbury/coppafish'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics'],
)