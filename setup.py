from setuptools import find_packages, find_namespace_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="magvit2",
    version="1.0",
    packages=find_namespace_packages(include=['taming.*']),
)
